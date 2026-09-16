import concurrent.futures
import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCausal(unittest.TestCase):
    def test_portable_entry_forwards_arguments_and_rejects_overwrite(self):
        m = load('profile_entry')
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / 'train.py'
            entry.touch()
            calls = []

            def run(path, run_name):
                calls.append((path, run_name, list(sys.argv)))

            original_argv = sys.argv
            original_path = list(sys.path)
            try:
                with (
                    mock.patch.dict(os.environ, {'RANK': '0', 'WORLD_SIZE': '16'}),
                    mock.patch.object(m, 'install') as install,
                    mock.patch.object(m, 'calibrate', side_effect=lambda p: dict(phase=p)),
                    mock.patch.object(m, 'receipt', return_value={}),
                    mock.patch.object(m.runpy, 'run_path', side_effect=run),
                ):
                    args = [
                        '--entry',
                        str(entry),
                        '--output-dir',
                        directory,
                        '--',
                        '--train-iters',
                        '8',
                    ]
                    m.main(args)
                    self.assertEqual(
                        calls, [(str(entry), '__main__', [str(entry), '--train-iters', '8'])]
                    )
                    self.assertTrue((Path(directory) / 'causal-clock-00000.json').is_file())
                    with self.assertRaises(FileExistsError):
                        m.main(args)
                    self.assertEqual(install.call_count, 1)
            finally:
                sys.argv = original_argv
                sys.path[:] = original_path

    def test_replay_advances_identity_without_graph_body_change(self):
        m = load('causal_ranges')
        events = []

        class NV:
            def range_push(self, x):
                events.append(x)

            def range_pop(self):
                events.append('pop')

        m.nvtx = lambda: NV()
        result = object()

        class Layer:
            layer_number = 29
            current_microbatch = 0

        calls = []

        def replay(self, token):
            calls.append(token)
            return result

        wrapped = m.replay_wrapper(replay)
        layer = Layer()
        token = object()
        m.STATE.step = 5
        for mb in (0, 1):
            layer.current_microbatch = mb
            self.assertIs(wrapped(layer, token), result)
        self.assertIs(calls[0], token)
        self.assertIs(calls[1], token)
        self.assertIn('/mb=0/layer=29/', events[0])
        self.assertIn('/mb=1/layer=29/', events[2])
        self.assertEqual(events.count('pop'), 2)
        self.assertIsNone(m.STATE.layer)

    def test_async_identity_bound_before_worker_and_no_batch_mutation(self):
        m = load('causal_ranges')

        class NV:
            def range_push(self, x):
                pass

            def range_pop(self):
                pass

        m.nvtx = lambda: NV()
        gate = threading.Event()

        class Iterator:
            def _prepare_batch(self, batch, *, item_idx):
                if item_idx == 0:
                    gate.wait(2)
                else:
                    gate.set()
                return batch

        iterator = Iterator()
        m.PREPARE_FN = Iterator._prepare_batch
        submit = m.submit_wrapper(concurrent.futures.ThreadPoolExecutor.submit)
        a = {'value': object()}
        b = {'value': object()}
        m.STATE.scheduling_iterator = iterator
        m.STATE.scheduling_window = '0:7'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fa = submit(pool, iterator._prepare_batch, a, item_idx=0)
            fb = submit(pool, iterator._prepare_batch, b, item_idx=1)
            self.assertIs(fa.result(), a)
            self.assertIs(fb.result(), b)
        self.assertEqual(set(a), {'value'})
        self.assertEqual(set(b), {'value'})
        self.assertEqual(m.BATCH_IDS[id(a)][1], ('0:7', 0))
        self.assertEqual(m.BATCH_IDS[id(b)][1], ('0:7', 1))

    def test_asymmetric_clock_bounds(self):
        m = load('clock_brackets')
        self.assertEqual(
            m.interval([[100, 117, 120, 110]]), dict(lower_ns=10, upper_ns=17, width_ns=7)
        )
        with self.assertRaises(AssertionError):
            m.interval([[100, 105, 120, 110]])

    def test_context_exception_and_nvtx_failures_restore(self):
        m = load('causal_ranges')
        for failure in ('native', 'push', 'pop'):

            class NV:
                def range_push(self, x):
                    if failure == 'push':
                        raise RuntimeError('push')

                def range_pop(self):
                    if failure == 'pop':
                        raise RuntimeError('pop')

            m.nvtx = lambda: NV()
            m.STATE.step = 99

            def native():
                self.assertEqual(m.STATE.step, 5)
                if failure == 'native':
                    raise RuntimeError('native')

            with self.assertRaises(RuntimeError):
                m.context(native, 'test', step=5)
            self.assertEqual(m.STATE.step, 99)

    def test_nested_thread_context_and_unrelated_submit(self):
        m = load('causal_ranges')

        class NV:
            def range_push(self, x):
                pass

            def range_pop(self):
                pass

        m.nvtx = lambda: NV()
        m.STATE.step = 99

        def inner():
            self.assertEqual(m.STATE.step, 6)

        def outer():
            self.assertEqual(m.STATE.step, 5)
            m.context(inner, 'inner', step=6)
            self.assertEqual(m.STATE.step, 5)

        m.context(outer, 'outer', step=5)
        self.assertEqual(m.STATE.step, 99)
        seen = []
        thread = threading.Thread(target=lambda: seen.append(getattr(m.STATE, 'step', None)))
        thread.start()
        thread.join()
        self.assertEqual(seen, [None])
        calls = []

        def submit(executor, fn, *args, **kwargs):
            calls.append((fn, args, kwargs))
            return 'sentinel'

        m.STATE.scheduling_window = '0:1'
        m.STATE.scheduling_iterator = object()

        def unrelated(value):
            return value

        self.assertEqual(m.submit_wrapper(submit)(object(), unrelated, 5), 'sentinel')
        self.assertIs(calls[0][0], unrelated)
        self.assertEqual(calls[0][1], (5,))

    def test_clock_solver_fail_closed_and_sensitivity(self):
        m = load('clock_bounds')
        self.assertEqual(
            m.host_shift({'x': 100}, [dict(name='x', before=110, after=112)]), (10, 12)
        )
        with self.assertRaises(AssertionError):
            m.host_shift({'x': 100}, [dict(name='x', before=110, after=2_000_000)])
        a = dict(bounds=dict(lower_ns=10, upper_ns=20), samples=[[0, 0, 0, 0]])
        b = dict(bounds=dict(lower_ns=30, upper_ns=40), samples=[[0, 0, 0, 0]])
        self.assertIsNone(m.offset_bounds([a, b], 1_000_000, 0))
        self.assertIsNotNone(m.offset_bounds([a, b], 1_000_000, 100))


if __name__ == '__main__':
    unittest.main()
