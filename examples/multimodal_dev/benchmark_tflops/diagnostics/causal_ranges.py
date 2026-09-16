"""Passive host NVTX identities, external batch identity map, no tensor operations."""

import concurrent.futures
import functools
import inspect
import os
import threading
import time

STATE = threading.local()
LOCK = threading.Lock()
BATCH_IDS = {}
CLOCK = []
TOPOLOGY = {}
COUNTS = {}
INSTALLED = False
ITERATOR_TYPE = None
PREPARE_FN = None


def nvtx():
    import torch

    return torch.cuda.nvtx


def context(function, label, **changes):
    old = {k: getattr(STATE, k, None) for k in changes}
    for k, v in changes.items():
        setattr(STATE, k, v)
    pushed = False
    try:
        nvtx().range_push('CAUSE/' + label)
        pushed = True
        return function()
    finally:
        try:
            if pushed:
                nvtx().range_pop()
        finally:
            for k, v in old.items():
                setattr(STATE, k, v)


def marker(name):
    before = time.monotonic_ns()
    nvtx().mark('CAUSE/CLOCK/' + name)
    after = time.monotonic_ns()
    CLOCK.append(dict(name=name, before=before, after=after))


def replay_wrapper(native, kind='layer_replay'):
    @functools.wraps(native)
    def wrapped(self, *args, **kwargs):
        layer = int(self.layer_number)
        microbatch = getattr(self, 'current_microbatch', None)
        step = getattr(STATE, 'step', None)
        label = f'{kind}/step={step}/mb={microbatch}/layer={layer}/ep={TOPOLOGY.get("ep")}'
        COUNTS[kind] = COUNTS.get(kind, 0) + 1
        return context(
            lambda: native(self, *args, **kwargs), label, layer=layer, microbatch=microbatch
        )

    return wrapped


def submit_wrapper(native):
    @functools.wraps(native)
    def wrapped(executor, fn, *args, **kwargs):
        window = getattr(STATE, 'scheduling_window', None)
        if (
            window is None
            or getattr(fn, '__self__', None) is not getattr(STATE, 'scheduling_iterator', None)
            or getattr(fn, '__func__', None) is not PREPARE_FN
        ):
            return native(executor, fn, *args, **kwargs)
        identity = (window, int(kwargs['item_idx']))

        # Closure is installed BEFORE executor.submit can start a worker.
        def work():
            result = context(
                lambda: fn(*args, **kwargs),
                f'materialize/window={window}/item={identity[1]}',
                window=window,
            )
            with LOCK:
                assert id(result) not in BATCH_IDS
                BATCH_IDS[id(result)] = (result, identity)
            return result

        return native(executor, work)

    return wrapped


def install():
    global ITERATOR_TYPE, PREPARE_FN, INSTALLED
    if INSTALLED:
        raise RuntimeError('causal instrumentation already installed')
    INSTALLED = True
    from examples.multimodal_dev.data.energon_mdp import MDPWindowMaterializingIterator as Iterator
    from examples.multimodal_dev.models.base import MultimodalModel
    from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from megatron.training import training

    ITERATOR_TYPE = Iterator
    PREPARE_FN = Iterator._prepare_batch
    native_step = training.train_step
    step_count = 0

    @functools.wraps(native_step)
    def step(*args, **kwargs):
        nonlocal step_count
        import torch

        step_count += 1
        if not TOPOLOGY:
            TOPOLOGY.update(
                rank=torch.distributed.get_rank(),
                ep=torch.distributed.get_process_group_ranks(
                    parallel_state.get_expert_tensor_and_model_parallel_group()
                ),
            )
            rank = TOPOLOGY['rank']
            expected = list(range(0, 8)) if rank < 8 else list(range(8, 16))
            assert TOPOLOGY['ep'] == expected, TOPOLOGY
        marker(f'step={step_count}/entry')
        try:
            return context(
                lambda: native_step(*args, **kwargs), f'step={step_count}', step=step_count
            )
        finally:
            marker(f'step={step_count}/exit')

    training.train_step = step
    TransformerLayer._te_cuda_graph_replay_impl = replay_wrapper(
        TransformerLayer._te_cuda_graph_replay_impl
    )
    TransformerLayer._te_cuda_graph_replay = replay_wrapper(
        TransformerLayer._te_cuda_graph_replay, 'layer_outer'
    )
    native_forward = schedules.forward_step
    signature = inspect.signature(native_forward)

    @functools.wraps(native_forward)
    def forward(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        mb = bound.arguments.get('current_microbatch')
        return context(
            lambda: native_forward(*args, **kwargs),
            f'forward_step/step={getattr(STATE,"step",None)}/mb={mb}',
            microbatch=mb,
        )

    schedules.forward_step = forward
    native_schedule = Iterator._schedule_window

    @functools.wraps(native_schedule)
    def schedule(self, *args, **kwargs):
        if not hasattr(self, '_causal_iterator_id'):
            self._causal_iterator_id = COUNTS.get('iterators', 0)
            COUNTS['iterators'] = self._causal_iterator_id + 1
        count = getattr(self, '_causal_window_counter', 0)
        self._causal_window_counter = count + 1
        window = f'{self._causal_iterator_id}:{count}'
        return context(
            lambda: native_schedule(self, *args, **kwargs),
            f'schedule/window={window}',
            scheduling_window=window,
            scheduling_iterator=self,
        )

    Iterator._schedule_window = schedule
    concurrent.futures.ThreadPoolExecutor.submit = submit_wrapper(
        concurrent.futures.ThreadPoolExecutor.submit
    )
    native_next = Iterator.__next__

    @functools.wraps(native_next)
    def consume(self, *args, **kwargs):
        result = native_next(self, *args, **kwargs)
        with LOCK:
            held, identity = BATCH_IDS.pop(id(result))
        assert held is result
        ids = getattr(STATE, 'consumed_ids', None)
        if ids is not None:
            ids.append(identity)
        nvtx().mark(
            f'CAUSE/consume_ready/step={getattr(STATE,"step",None)}/window={identity[0]}/item={identity[1]}'
        )
        return result

    Iterator.__next__ = consume
    native_sidecar = MultimodalModel.pipeline_sidecar_pre_forward
    sidecar_signature = inspect.signature(native_sidecar)

    @functools.wraps(native_sidecar)
    def sidecar(self, *args, **kwargs):
        bound = sidecar_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        mb = bound.arguments.get('current_microbatch')
        return context(
            lambda: native_sidecar(self, *args, **kwargs),
            f'sidecar/step={getattr(STATE,"step",None)}/mb={mb}',
            consumed_ids=[],
            pack=0,
            microbatch=mb,
        )

    MultimodalModel.pipeline_sidecar_pre_forward = sidecar
    native_vision = Qwen35VLVisionEncoder.forward

    @functools.wraps(native_vision)
    def vision(self, *args, **kwargs):
        pack = getattr(STATE, 'pack', None)
        if pack is None:
            return native_vision(self, *args, **kwargs)
        STATE.pack = pack + 1
        ids = list(getattr(STATE, 'consumed_ids', []) or [])
        label = f'vision_pack/step={getattr(STATE,"step",None)}/mb={getattr(STATE,"microbatch",None)}/pack={pack}/consumed_window_context={ids}'
        return context(lambda: native_vision(self, *args, **kwargs), label)

    Qwen35VLVisionEncoder.forward = vision


def receipt():
    return dict(
        topology=TOPOLOGY,
        clock_markers=CLOCK,
        counts=COUNTS,
        remaining_prefetched_items=len(BATCH_IDS),
        pid=os.getpid(),
    )
