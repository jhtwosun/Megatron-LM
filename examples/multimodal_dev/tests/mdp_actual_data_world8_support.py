# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared storage-backed Energon-document fixtures for repeated-D4 world8 tests."""

from __future__ import annotations

import io
import os
import pickle
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import torch
import torch.distributed as dist
from PIL import Image

from examples.multimodal_dev import forward_step
from megatron.core.mdp import integration
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.runtime import MdpRuntimeState


@dataclass(frozen=True)
class ActualDataFixture:
    parent: str
    root: str

    def descriptors(self, model_arch: str, domain_lane: int):
        size = 256 if model_arch == "qwen35_vl" else 32
        grid = (1, size // 16, size // 16)
        directory = Path(self.root) / model_arch
        common = {"grid_thw": grid, "height": size, "width": size}
        if domain_lane == 0:
            return (
                {"kind": "image_path", "path": str(directory / "plain.jpg"), **common},
                {
                    "kind": "zip_image",
                    "zip_path": str(directory / "images.zip"),
                    "candidate": "nested/image.jpg",
                    **common,
                },
            )
        return (
            {
                "kind": "parquet_column_image",
                "parquet_path": str(directory / "table" / "images.parquet"),
                "column": "image",
                "row_idx": 0,
                **common,
            },
            {
                "kind": "jpgs",
                "path": str(directory / "images.jpgs"),
                "encoded_image_index": 1,
                **common,
            },
        )


class RecordingAllocator(DirectBufferAllocator):
    """Record only allocation metadata; buffer ownership remains native."""

    def __init__(self):
        super().__init__()
        self.events = []

    def acquire(self, *, rows, width, dtype, device, tag):
        tensor = super().acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)
        self.events.append((tag, rows, width, tensor))
        return tensor


def assert_world_check(check):
    """Turn a rank-local oracle failure into one identical WORLD assertion."""
    failure = None
    try:
        check()
    except BaseException as error:
        failure = (type(error).__name__, str(error))
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    assert failures == [None] * dist.get_world_size(), failures


def install_protocol_observers(monkeypatch, *, dynamic_decoder: bool):
    """Observe the active facade without replacing any protocol operation."""
    if dynamic_decoder:
        from megatron.core.mdp import dynamic_cp_d4_joint_facade as facade
    else:
        from megatron.core.mdp import dynamic_cp_d4_fixed_facade as facade

    observed = {
        "projections": [],
        "authorities": [],
        "claims": [],
        "publications": [],
        "broadcasts": [],
    }
    native_gather = facade._gather_d4_source_catalog
    native_build = facade.build_repeated_d4_joint_iteration_authority
    native_forward = facade._forward.run_repeated_d4_encoder_forward
    native_publication = facade._forward.run_repeated_d4_encoder_publication

    def gather(*args, **kwargs):
        projection = native_gather(*args, **kwargs)
        observed["projections"].append(projection)
        return projection

    def build(*args, **kwargs):
        authority = native_build(*args, **kwargs)
        observed["authorities"].append(authority)
        return authority

    def run_forward(runtime, claim, authority, **kwargs):
        observed["claims"].append(
            (claim.capture_mode, claim.locator_catalog, tuple(claim.pixel_sidecar.items()))
        )
        native_broadcast = kwargs.get("broadcast", dist.broadcast)

        def broadcast(tensor, **broadcast_kwargs):
            observed["broadcasts"].append((tensor.dtype, tuple(tensor.shape)))
            return native_broadcast(tensor, **broadcast_kwargs)

        kwargs["broadcast"] = broadcast
        return native_forward(runtime, claim, authority, **kwargs)

    def publish(owner, **kwargs):
        observed["publications"].append(tuple(owner.item_outputs))
        return native_publication(owner, **kwargs)

    monkeypatch.setattr(facade, "_gather_d4_source_catalog", gather)
    monkeypatch.setattr(facade, "build_repeated_d4_joint_iteration_authority", build)
    monkeypatch.setattr(facade._forward, "run_repeated_d4_encoder_forward", run_forward)
    monkeypatch.setattr(facade._forward, "run_repeated_d4_encoder_publication", publish)
    return observed


def _jpeg_bytes(size: int, color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (size, size), color).save(stream, format="JPEG", quality=100, subsampling=0)
    return stream.getvalue()


def create_shared_fixture() -> ActualDataFixture:
    """Create one rank0-owned fixture tree on an explicitly shared filesystem."""
    configured = os.environ.get("MDP_I1C_FIXTURE_ROOT")
    if not configured:
        raise RuntimeError("MDP_I1C_FIXTURE_ROOT must name an explicit shared parent")
    parent = Path(configured)
    if (
        not parent.is_absolute()
        or str(parent) != os.path.normpath(configured)
        or str(parent) in ("/", "/project", "/lustre", "/mnt")
        or not str(parent).startswith(("/project/", "/lustre/", "/mnt/"))
    ):
        raise RuntimeError("MDP_I1C_FIXTURE_ROOT must be a confined canonical shared parent")
    status = [None]
    if dist.get_rank() == 0:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            parent.mkdir(parents=True, exist_ok=True)
            if parent.is_symlink() or parent.resolve() != parent:
                raise RuntimeError("I1c shared fixture parent must not contain a symlink alias")
            root = parent / f"mdp-i1c-{uuid4().hex}"
            root.mkdir()
            for model_arch, size in (("qwen35_vl", 256), ("nemotron_omni", 32)):
                directory = root / model_arch
                table = directory / "table"
                table.mkdir(parents=True)
                first = _jpeg_bytes(size, (31, 97, 173))
                second = _jpeg_bytes(size, (173, 59, 41))
                (directory / "plain.jpg").write_bytes(first)
                (table / "payload.jpg").write_bytes(second)
                (directory / "images.jpgs").write_bytes(pickle.dumps([first, second], protocol=4))
                with zipfile.ZipFile(directory / "images.zip", "w") as archive:
                    archive.writestr("nested/image.jpg", second)
                pq.write_table(
                    pa.table({"image": [{"path": "payload.jpg"}]}), table / "images.parquet"
                )
            status[0] = ("ok", str(root))
        except BaseException as error:
            status[0] = ("error", type(error).__name__, str(error))
    dist.broadcast_object_list(status, src=0)
    if type(status[0]) is not tuple or not status[0] or status[0][0] != "ok":
        raise RuntimeError(f"rank0 could not create I1c shared fixture: {status[0]}")
    root = status[0][1]
    dist.barrier()
    return ActualDataFixture(str(parent), root)


def remove_shared_fixture(fixture: ActualDataFixture) -> None:
    dist.barrier()
    if dist.get_rank() == 0:
        parent = Path(fixture.parent)
        root = Path(fixture.root)
        if (
            root.parent != parent
            or not root.name.startswith("mdp-i1c-")
            or root.is_symlink()
            or root.resolve().parent != parent.resolve()
        ):
            raise RuntimeError("refusing to remove an untrusted I1c fixture path")
        shutil.rmtree(root)
    dist.barrier()


def launch_args(fixture: ActualDataFixture, model_arch: str):
    image_token = 248056 if model_arch == "qwen35_vl" else 18
    return SimpleNamespace(
        dataset_provider="energon",
        energon_path=fixture.root,
        model_arch=model_arch,
        mdp_enable=True,
        mdp_vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
        use_packed_sequence=True,
        seq_length=512 if model_arch == "qwen35_vl" else 64,
        sequence_parallel=False,
        image_token_id=image_token,
        vision_spatial_merge_size=2,
        fp4=None,
        fp8=None,
        fp8_recipe=None,
    )


def storage_document(fixture: ActualDataFixture, model_arch: str, rank: int):
    lane = rank // 4
    descriptors = fixture.descriptors(model_arch, lane)
    grids = tuple(descriptor["grid_thw"] for descriptor in descriptors)
    image_token = 248056 if model_arch == "qwen35_vl" else 18
    output_rows = tuple(t * (h // 2) * (w // 2) for t, h, w in grids)
    tokens = [11]
    for rows in output_rows:
        tokens.extend([image_token] * rows)
    tokens.append(12)
    input_ids = torch.tensor(tokens, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[-1] = 12
    loss_mask = torch.zeros(input_ids.shape, dtype=torch.float32)
    loss_mask[-1] = 1
    payload_width = 1536 if model_arch == "qwen35_vl" else 768
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "pixel_values": torch.empty((0, payload_width), dtype=torch.float32),
        "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
        "image_descriptors": descriptors,
    }


def text_document(model_arch: str):
    payload_width = 1536 if model_arch == "qwen35_vl" else 768
    return {
        "input_ids": torch.tensor((11, 12), dtype=torch.long),
        "labels": torch.tensor((-100, 12), dtype=torch.long),
        "loss_mask": torch.tensor((0.0, 1.0)),
        "pixel_values": torch.empty((0, payload_width), dtype=torch.float32),
        "image_grid_thw": torch.empty((0, 3), dtype=torch.int64),
        "image_descriptors": (),
    }


def run_text_only_public(
    monkeypatch,
    *,
    runtime,
    fixture,
    model_arch,
    dynamic_decoder,
    decoder,
    adapter_module,
    native_bind,
):
    """Run one locator-mode text document and prove the storage path stays dormant."""
    from examples.multimodal_dev.data.energon import materializer as generic

    integration._RUNTIME = runtime
    monkeypatch.setattr(forward_step, "get_args", lambda: launch_args(fixture, model_arch))
    reads = []
    binds = []
    monkeypatch.setattr(
        adapter_module,
        "_bind_dynamic_encoder_cp",
        lambda *args, **kwargs: binds.append(kwargs["membership"].group_size)
        or native_bind(*args, **kwargs),
    )
    monkeypatch.setattr(
        generic,
        "vision_locator_image_bytes",
        lambda locator: reads.append(locator)
        or (_ for _ in ()).throw(AssertionError("text-only locator execution performed image I/O")),
    )
    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        min_dynamic_context_parallel_size=1,
        max_seqlen_per_dp_cp_rank=16,
        finalize_model_grads_func=lambda _model, tokens: dist.all_reduce(tokens),
    )
    saw_leaf = []
    native_forward = decoder.forward

    def observe_forward(**kwargs):
        saw_leaf.append(kwargs["vision_embeddings"])
        return native_forward(**kwargs)

    monkeypatch.setattr(decoder, "forward", observe_forward)

    def native_schedule(data_iterator, num_microbatches, forward_only):
        assert num_microbatches == 1 and forward_only is False
        output, output_loss_func = forward_step.forward_step(data_iterator, decoder)
        loss, tokens, _ = output_loss_func(output)
        loss.backward()
        config.finalize_model_grads_func([], tokens)
        return loss.detach()

    wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
    loss = wrapped(
        data_iterator=iter(([text_document(model_arch)],)), num_microbatches=1, forward_only=False
    )
    assert torch.isfinite(loss)
    assert reads == [] and binds == [] and saw_leaf == [None]
    assert not any(
        event[0] in ("dynamic_cp_gate0_locator_pixels", "dynamic_cp_gate0_pixels")
        for event in runtime.allocator.events
    )
    assert runtime.allocator._outstanding == 0
    assert runtime.storage.get_leaf(0) is None
    assert runtime.iteration == 1 and runtime.state is MdpRuntimeState.EMPTY
    return loss


def run_actual_locator_parity_arm(
    monkeypatch,
    *,
    runtime,
    fixture,
    model_arch,
    dynamic_decoder,
    decoder,
    adapter_module,
    native_bind,
    expected_encoder_cp,
):
    """Run one fresh actual-storage arm and return output and both-domain gradients."""
    integration._RUNTIME = runtime
    monkeypatch.setattr(forward_step, "get_args", lambda: launch_args(fixture, model_arch))
    observed_cp = []

    def observe_bind(*args, membership, **kwargs):
        observed_cp.append(membership.group_size)
        return native_bind(*args, membership=membership, **kwargs)

    monkeypatch.setattr(adapter_module, "_bind_dynamic_encoder_cp", observe_bind)
    decoder_vision = []
    native_decoder_forward = decoder.forward

    def observe_decoder_forward(**kwargs):
        decoder_vision.append(kwargs["vision_embeddings"].detach().float().clone())
        return native_decoder_forward(**kwargs)

    monkeypatch.setattr(decoder, "forward", observe_decoder_forward)
    runtime.encoder_domain.encoder_ddp.zero_grad_buffer()
    decoder.zero_grad(set_to_none=True)
    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        min_dynamic_context_parallel_size=1,
        max_seqlen_per_dp_cp_rank=128 if model_arch == "qwen35_vl" else 16,
        finalize_model_grads_func=lambda _model, tokens: dist.all_reduce(tokens),
    )

    def native_schedule(data_iterator, num_microbatches, forward_only):
        assert num_microbatches == 1 and forward_only is False
        output, output_loss_func = forward_step.forward_step(data_iterator, decoder)
        loss, tokens, _ = output_loss_func(output)
        loss.backward()
        config.finalize_model_grads_func([], tokens)
        return loss.detach()

    try:
        wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
        document = storage_document(fixture, model_arch, dist.get_rank())
        loss = wrapped(data_iterator=iter(([document],)), num_microbatches=1, forward_only=False)
        encoder_grads = {
            name: parameter.main_grad.detach().float().clone()
            for name, parameter in runtime.encoder_domain.encoder_ddp.named_parameters()
            if parameter.main_grad is not None
        }
        decoder_grads = {
            name: parameter.grad.detach().float().clone()
            for name, parameter in decoder.named_parameters()
            if parameter.grad is not None
        }
        selected = dist.get_rank() % 4 < expected_encoder_cp
        activity = [None] * dist.get_world_size()
        dist.all_gather_object(
            activity,
            (
                any(torch.count_nonzero(value) for value in encoder_grads.values()),
                any(torch.count_nonzero(value) for value in decoder_grads.values()),
            ),
        )

        def validate_arm():
            assert observed_cp == ([expected_encoder_cp] if selected else []), observed_cp
            assert encoder_grads and decoder_grads, (encoder_grads.keys(), decoder_grads.keys())
            assert all(value[0] for value in activity), activity
            decoder_activity = [value[1] for value in activity]
            assert decoder_activity[:4] == decoder_activity[4:], activity
            assert sum(decoder_activity[:4]) == 1, activity
            assert len(decoder_vision) == 1, len(decoder_vision)

        assert_world_check(validate_arm)
        return loss.float(), encoder_grads, decoder_grads, decoder_vision[0]
    finally:
        integration.reset_for_testing()


def assert_actual_gradients_close(actual, reference, *, rtol):
    """Compare full gradient mappings while admitting jointly-zero parameters."""
    assert actual.keys() == reference.keys()
    for name in actual:
        candidate = actual[name].float()
        baseline = reference[name].float()
        assert candidate.shape == baseline.shape
        assert torch.isfinite(candidate).all() and torch.isfinite(baseline).all()
        baseline_norm = float(baseline.norm())
        if baseline_norm == 0:
            assert torch.count_nonzero(candidate) == 0, name
            continue
        candidate_norm = float(candidate.norm())
        relative_l2 = float((candidate - baseline).norm()) / baseline_norm
        cosine = float(candidate.flatten().dot(baseline.flatten())) / (
            candidate_norm * baseline_norm
        )
        diagnostic = f"{name}: relative_l2={relative_l2}, cosine={cosine}"
        assert relative_l2 <= rtol and cosine >= 0.999, diagnostic
