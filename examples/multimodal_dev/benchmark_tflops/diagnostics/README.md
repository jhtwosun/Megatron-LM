# Opt-in arrival diagnostics

This additive host-side instrumentation helps distinguish a long collective
from late participant arrival. It does not optimize or modify native training
code, tensor values, batch schemas, CUDA synchronization, or graph bodies.
It is separate from the TFLOPs estimators in the parent directory.

## Supported scope and integration

The retained experiment used W16 TP1 PP2 CP2 EP8 ETP1, GBS64 MBS1,
S8192, BF16, 8 training steps and captured steps 5–7. It used the PR7
full Qwen3-VL configuration (using retained qwen35_vl implementation modules),
fused vision recompute cap131072,
decoder recompute OFF and TE attention/router/preprocess graphs. This wrapper
deliberately checks world16 and EP groups 0–7 / 8–15; it is not a generic
topology profiler. Numbered train_step calls assume no rerun/resume/evaluation.

The historical run additionally required an external Bridge data entry,
canonical remuxed data and the original qualified environment. Those are NOT
included here. A checkout of this PR alone does not reproduce that real-data
run, and this portable wrapper has CPU tests, not a new GPU qualification.
The historical runtime imported separate PAIR ranges and an environment-specific
NVTX import compatibility fix; neither is silently injected by this wrapper.
The CAUSE ranges here identify original sidecar vision packs and layer replay;
they do not supply every legacy PAIR phase or recompute classification.

Use an already qualified training entry, arguments and environment. On each
torchrun rank, replace only its Python entry with this wrapper:

```bash
export REPO_ROOT=/path/to/qualified/Megatron-LM
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
export CAUSAL_CLOCK_PORT=29650 # reserve this and the next TCP port
python examples/multimodal_dev/benchmark_tflops/diagnostics/profile_entry.py \
  --entry /path/to/qualified_training_entry.py \
  --output-dir /path/to/new/existing/receipt_directory -- <unchanged training args>
```

The repository import path must be available before hook installation, which
precedes execution of the supplied training entry. Preserve any additional
PYTHONPATH entries required by that qualified entry and environment.
The launcher must set RANK, WORLD_SIZE and MASTER_ADDR, mount identical source
and shared output paths, and permit rank0 TCP connections. Each rank writes one
exclusive causal-clock-NNNNN.json after successful training. No success receipt
is emitted after a training exception. Do not reuse output directories.
No scheduler command is provided: use the site's allocated diagnostic workflow,
not a login node. Wrap the rank process with Nsight Systems NVTX/CUDA capture
and retain all rank traces. Keep native emit_nvtx enabled and original capture
controls; the wrapper itself does not start/stop CUDA profiling.

## Identity and clock semantics

- Layer/microbatch identities wrap the actual TE replay invocation, not only
  graph capture. Native ordered EP membership is recorded once after init.
- Window/item identity is bound before executor workers start. Batch objects
  are retained in an external side map until consumption, without schema edits.
  Unconsumed prefetch references remain until process exit and are counted.
- consume_ready is an instant event: retain it when exporting NVTX. A pack's
  consumed_window_context is cumulative whole-window context, not exact image
  membership. Materialization outside capture is unavailable, never zero.
- CPU ping-pong before warmup and after training bounds offset without symmetric
  latency assumptions: [t3-t4, t2-t1]. Each phase has a 120-second deadline.
  Step NVTX markers bracket trace-to-host monotonic conversion.
- clock_bounds.py provides pure interval helpers. Pre/post offset intervals
  alone do not certify intervening clock drift; report an explicit sensitivity
  bound and reject disjoint intervals. Do not align clocks by forcing NCCL ends
  equal. CPU range return and collective enqueue are not GPU input readiness.

For analysis, join CUDA launches by process/correlation identity, preserve
stream/event dependencies and unknown graph attribution, match exact
(step, microbatch, global layer, EP group), and report a candidate family when
clock intervals cannot distinguish CP pair members. Do not infer expert IDs
from grouped-GEMM kernel names. Full historical extraction/solver scripts and
private raw traces are not shipped as a portable end-to-end analyzer here.

## CPU tests

```bash
python3 -m unittest discover -s examples/multimodal_dev/benchmark_tflops/diagnostics -p 'test_*.py' -v
```

Tests use stub NVTX and standard-library threads, no Torch or GPU imports.
See [historical findings](FINDINGS.md) for measured evidence and remaining gaps.
