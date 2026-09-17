# Packed vision self-attention: first-use plan preparation

This standalone diagnostic isolates the TE/cuDNN packed self-attention forward
used by the vision workload discussed in PR #131. It does **not** require
Megatron imports, model weights, a dataset, a dataloader, NCCL or multiple GPUs.
It does not change training code or claim an optimization.

## Run

Use a CUDA GPU with a compatible PyTorch/Transformer Engine installation.
The reproduction target is the original TE 2.18 environment; this script uses
TE's version-dependent `cpp_extensions.fused_attn` API. Do not run it on a
cluster login node. Launch on an allocated compute node:

```bash
CUDA_VISIBLE_DEVICES=0 CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=cudnn.log \
  python examples/multimodal_dev/diagnostics/attention_plan_repro.py > calls.jsonl
```

Run each invocation in a fresh process; do not overwrite prior backend logs.
Backend logging is diagnostic overhead, enabled identically for every shape.
The script explicitly calls the BF16 arbitrary-sequence fused backend, with no
FlashAttention/unfused fallback. Unsupported environments should fail visibly.
It prints versions, device, PID/TID and per-call JSON measurements.

For a GUI trace, optionally wrap the same command with
`nsys profile --trace=cuda,nvtx --output=attention_plan`. Inspect NVTX
`packed_attention/<case>/visit=<n>` ranges and their host/GPU work separately.
Nsight profiling is optional, not required to execute this script.

Standard-library shape checks (no Torch import):

```bash
python examples/multimodal_dev/diagnostics/test_attention_plan_repro.py
```

## What is held fixed and what changes

BF16, THD, 16 heads, head dimension128, training=True, dropout0, noncausal
padding mask, separate contiguous Q/K/V and max sequence length4096 are fixed
for A/B/C. Storage is `[sum(lengths),16,128]`, **not** a dense bucket-padded
`[B,T_bucket,16,128]` allocation. Inputs and cumulative lengths are prepared
outside all timed calls.

| Case | Sequence lengths | B | Total tokens | Expected TE token-capacity bucket |
|---|---|---:|---:|---:|
| Warmup | 256 |1|256|1024|
| A |4096 + eight2048|9|20480|32768|
| B |4096 + seven2048 + two1024|10|20480|32768|
| C |nine4096|9|36864|65536|

The default order is warmup twice, A three times, B three times, A again,
C three times, then A/B/C again. A/B change B while preserving total tokens
and max length; A/C change total-token capacity at fixed B and max length.
A/B have different sums of squared lengths: they are **not** equal-attention-
FLOP performance comparisons. First/repeated calls **within each case** reuse
identical tensors and are the relevant latency comparison.

Expected bucket values are hypotheses to check in the actual backend log,
not proof of which compiled-library path was used. Inspect printed Q
dimensions and execution-plan Finalize/Execute records. Older cuDNN/TE paths
may bucket B differently; record that rather than forcing the expected result.

## Timing and validation

- `host_call_ms`: time until the TE Python call returns; includes native plan
  preparation, dispatch and output/workspace allocation, not only plan build.
- `completed_wall_ms`: same start through CUDA synchronization; includes host
  preparation and device completion. It is **not pure GPU kernel time**.
- CUDA events around a Python call can include device idle while the host
  prepares a plan. This script intentionally does not label them GPU compute.
- Output finite checks and exact same-input repeated-output checks execute
  outside timing. These are forward sanity checks, not a reference SDPA or
  backward/gradient parity test. `training=True` requests training auxiliaries;
  no backward is run.

To attribute a first-call delay to plan preparation, match PID/TID, call order,
actual descriptors, plan Finalize and subsequent Execute records. A slow first
call alone does not prove compilation. Do not call the entire host interval
JIT time, kernel time, or autotuning. Repeated-shape reuse after other shapes
helps distinguish shape-local preparation from generic process initialization.

## Connection to the original observation

The four-rank full-model diagnostic in PR #131 observed14 slow initial native
workspace queries with execution-plan Finalize records and22 fast reused ones
without those records. Finalize descriptor-to-success timestamp separations
were approximately794–927ms. These are historical logged intervals, not exact
internal API/JIT duration measurements or a promised latency on other systems.

Rank-local first use matters: a plan already used by one encoder rank can be
new on another. Variable real-data packing changes B and token buckets even
at a fixed pack cap, so FLOP-balanced owners can still reach embedding exchange
at different times. Early decoder ranks may then wait in collectives. The
single-GPU reproduction isolates the attention preparation component only;
it does not reproduce distributed waits or measure end-to-end throughput.

Source reference: TE v2.18 forward plan lookup/build in
[fused_attn_f16_arbitrary_seqlen.cu](https://github.com/NVIDIA/TransformerEngine/blob/27486e03cfc1fa41f6932dcecdc47c71c47eac3e/transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu).
The function-local thread-local map is per process/thread, not shared among
training ranks. Matching installed binary behavior requires the runtime
descriptor/Finalize evidence, not just reading this source.

## Reproduction results

Reproduced on one GB200 in interactive allocation425984, step0 completed0
(2026-09-16 PDT). PyTorch2.13.0a0+9186a08b2c.nv26.07, CUDA13.3,
Transformer Engine2.18.0, cuDNN9.25.0. One process/thread, no distributed
initialization, no full model or dataset. All15 forward calls completed with
finite outputs; all11 repeated calls matched the corresponding first output
exactly. Shape-contract tests:3 passed. Independent code/runtime review passed.

| Case | First host call ms | Reused host median ms | First completed wall ms | Reused completed wall median ms |
|---|---:|---:|---:|---:|
| A |824.438|0.457|824.697|0.741|
| B |821.470|0.450|821.727|0.732|
| C |820.784|0.427|821.562|1.230|

Repeated counts are A4/B3/C3. The unrelated warmup's first call was1320.297ms;
it is excluded from the table because it also includes generic initialization.
The table is one diagnostic process, not a repeated-trial performance estimate.
**The approximately820ms penalty persists for novel shapes after warmup.**

The full backend log contains exactly4 execution-plan Finalize descriptor
records (warmup/A/B/C) and15 Execute descriptor records, all in the same PID/TID.
Actual Q descriptors match the table's expected buckets exactly. Each new
shape has one plan Finalize before its first Execute; no repeated shape has
another plan Finalize, including revisits after different shapes. All select
engine10. This supports shape-local plan preparation rather than a slow
steady-state self-attention kernel. It does not isolate internal JIT duration,
prove that all attention latency comes from preparation, or show a model fix.

Traceability (SHA256):

- TE common library, same as the original full-model diagnostic:
  `57bed517cf34c63d7236d1b549622ff034fa924fc8273edae17689d0e9dd4dae`.
- cuDNN dispatcher library:
  `ba43ba0d2980706c140f7d99fd107ea9cbcb4cc4d6b3d691cae8a24d351e6a0a`.
- Raw backend log:
  `7f22acdff1e5cca7492d2fdff26407ad2d292aac10755b06e0a840f409cc0b2b`.

The raw backend log is approximately7.5MiB and is not committed. The commands
above generate a fresh one; timings need not match these absolute values on
other versions/devices. Library hashes identify the tested TE common and
cuDNN dispatcher, not every library loaded into the process.
