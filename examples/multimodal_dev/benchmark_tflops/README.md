# PR7 workload-aware TFLOPs reproduction (mock and real Energon)

This is an experimental reproduction snapshot stacked on
[PR7](https://github.com/jhtwosun/Megatron-LM/pull/7),
base `e1484af4f5e9e5723105f731fb555dba9a32fecb`.
It includes the runtime changes needed by the measured configuration, not just
a reporting formula. There is no cleanup/refactor of the legacy MDP sidecar.
This is **not** the BestJuly MDP runtime or a general model FLOPs calculator.

## What the numbers mean

Three independently labeled estimates share the same **whole multimodal
optimizer-step wall-time** denominator:

1. Native `throughput per GPU (TFLOP/s/GPU)`: unchanged MCore fixed-shape
   decoder FLOPs, based on `GBS * seq_length` and full-bin attention.
2. Useful packed-content decoder FLOPs: actual retained content lengths and
   within-document attention only; padding is excluded.
3. Useful vision FLOPs: actual global image/video grids. Report separately,
   or add to (2) for the combined useful matmul estimate.

Do **not** add vision to (1) and call it useful FLOPs. A decoder-only numerator
over VLM time is **not** a decoder-only timed benchmark. These are analytic
estimates, not hardware counters, and a larger reported number alone does not
establish faster execution.

### Geometry and normalization

Before padding and owner-local image materialization, each packed bin records:

| Counter | Definition | Work represented |
|---|---|---|
| T | sum of original retained document lengths | Decoder content, including visual embedding tokens |
| U | sum of each document length squared | Within-document causal attention |
| R | sum of t*h*w over global image grids | Vision patch rows before spatial merge |
| A | sum of t*(h*w)^2 | Per-frame bidirectional vision attention |

Mock obtains these from the reference scenario adapter before native packing
alignment. Real Energon uses original `content_lens` and global grids in
`task_encoder.py`, before owner prepartition. These are not supervised-token
counts, not local pixel rows summed across replicated ranks, and not the square
of the packed bin length. Temporal grids use t*(h*w)^2, not (t*h*w)^2.

The four `ref_geom_T/U/R/A` fields travel through the existing reporting
reduction as int64 count/one pairs. Native microbatch/DP/CP averaging yields a
mean per logical packed bin. Multiply each logged mean by GBS to recover global
step geometry. Float64 logging with 12E precision reduces rounding error; it is
still not an exact integer archive. No loss/backward normalization is changed.

### Model formula

See [workload_flops.py](workload_flops.py) for the executable formula.
FMA=2, training=3*forward, causal attention uses the half-square convention.
For the **full PR7 Qwen3-VL** model:

- Decoder L=48, H=2048, Q projection=4096, KV projection=512, 128 experts,
  top-k=8, expert FFN=768, no shared expert, padded vocab=248448.
- Vision L=27, H=1152, FFN=4304, patch input=1536, spatial merge=4,
  attention head dimension=72. Attention-only 72-to-128 kernel padding does
  not increase useful model FLOPs.

Forward decoder components:
`2*L*T*H*(Q+2*KV) + 2*L*U*Q + 2*L*T*Q*H + 6*L*T*K*H*F + 2*T*H*V`.

Forward vision components:
`2*R*1536*Hv + 8*Lv*R*Hv^2 + 4*Lv*A*Hv + 4*Lv*R*Hv*Fv
 + 2*(R/4)*((4*Hv)^2 + (4*Hv)*H)`.

Multiply forward sums by 3. TFLOPs/GPU =
`training_flops / (world_size * step_ms * 1e9)`.
Router, norm, RoPE, softmax, optimizer, padding, and internal kernel recomputation
are excluded. Recompute is OFF in the reported experiment. Changing model,
vocabulary, vision tower, MTP or recompute requires a new validated formula;
changing GPU count alone only changes the denominator.

### Timing boundary and windows

An iteration is one optimizer update, not one microbatch forward. The logged
iteration timer includes in-step loading/waiting, encoder/decoder execution,
communication and optimizer work. Asynchronous prefetch overlapped with compute
is not an extra additive time term. This does not include all process startup,
environment setup or complete dataset construction time.

Run 20 total optimizer steps (`18 + 2` launcher convention). Reference-style
report: iterations **4–20**, 17 samples; exclude graph capture/warmup. Also
report **10–20**, 11 samples, separately for the existing campaign convention.
The portable collector provides mean and median, never mixes their definitions,
and rejects incomplete/duplicate/nonfinite/skipped/NaN rows or a GBS mismatch.
It does not itself certify scheduler exit status, exact source, or backend.
Record job exit0 and artifacts separately. Do not label 10–20 as 10–50.

## Locked experiment configuration

| Setting | Value |
|---|---|
| Model / precision | Full PR7 Qwen3-VL, random init, BF16, no checkpoint |
| World / topology | 16 GB200: TP1 PP2 CP2 EP8 ETP1, dense DP4 |
| Batches | MBS1 GBS64, 16 microbatches per optimizer step |
| Decoder budget | 16384 tokens per packed bin |
| MDP | Original PR7 sidecar, owner prepartition, PPxCP scope |
| Vision | Fused retain, sequence cap131072 **patch tokens**, not pixels |
| Recompute | Decoder OFF, whole-encoder OFF |
| Other | MTP0 VPP0, HybridEP, forced-balanced router |
| Graphs | attn + moe_router + moe_preprocess, warmup2 |
| Optimizer | Precision-aware distributed Adam |
| Other tuning | HybridEP chunks128, manual GC10, norm SM margins0 |

Legacy MDP disables gradient-reduce and parameter-gather overlap before DDP
construction; preparse flags do not prove overlap is enabled. MoE A2A overlap is OFF.
Do not silently enable recompute, change cap, use a proxy model, or change
topology after OOM and label it the same cell.

### Workload differences

**Mock:** reference lognormal scenario distribution, min512/max4096,
mean parameter2048, sigma1.1, dynamic image/text geometry, native 64-token
document alignment, workers0. Synthetic supervision is not real HF supervision.
Static metadata capacity32 = up to31 real documents plus one masked dummy tail
(33 cumulative-length endpoints).

**Real:** equal-weight Mantis/M4/Pixmo lazy Energon blend3, actual local HF
tokenizer/chat-template assistant masks, image min0/max327680 pixels,
packing/shuffle buffers128, workers1/prefetch1, original greedy token-budget
packing. The cap131072 above is a different unit from this pixel budget.
`energon-max-samples-per-sequence=16` controls Energon shard slicing, **not**
the number of documents packed into a decoder bin.
The example preserves `subflavors.crude_type: qwen35`; direct dataset metadata
may say `qwen35_lazy`, which is not the same cooker override.
Inspect the existing [data provider](../data/qwen35_energon/provider.py) and
[task encoder](../data/qwen35_energon/task_encoder.py) for the expected schema.
These are prepared lazy datasets, not arbitrary JPEG directories.

Real static metadata capacity129 = up to128 real documents plus one dummy tail
(130 endpoints), matching the hard packing-buffer candidate bound. It does not
truncate samples or change greedy packing. Original real boundaries are retained;
a separate zero-loss dummy tail replaces extending the final real segment to
the full bin budget. This changes graph metadata shape and must be qualified.
Real and mock are not matched-sample causal comparisons.

## Run on another cluster

### Environment

Use a separately qualified CUDA/Torch/TE environment on the target architecture.
The measured ARM64 stack was:

- Torch `2.13.0a0+9186a08b2c.nv26.07`, CUDA13.3.
- TE2.18.0, cuDNN9.25.0.15, cuBLASLt13.6.0.2.
- FA4 `4.0.0b11`, CUTLASS DSL4.4.2.
- HybridEP `7febc6e25660af0f54d95dd781ecdcd62265ecca`, rebuilt against this Torch.
- Transformers4.57.1, tokenizers0.22.2, huggingface-hub0.36.2.
- Real adds Energon7.3.2; supplemental installed versions are recorded in
  [real-overlay-versions.txt](real-overlay-versions.txt), **not** a complete lock.

Do not copy the ARM64 virtualenv or compiled HybridEP/TE binaries onto x86.
Resolve/install from appropriate releases/source and qualify on allocated GPUs.
If using an overlay, set ENERGON_OVERLAY without mutating the qualified
Torch/TE prefix. Set PATH to the intended venv, both CUDNN_HOME and CUDNN_PATH
to its cuDNN prefix, and prepend its lib directory to LD_LIBRARY_PATH.
The native launcher uses `python -m torch.distributed.run`, avoiding a global
torchrun shebang that could select another interpreter.

Our base OS supplied cuBLAS despite a pip missing-distribution warning from
cuDNN; that metadata exception was checked against actual loaded libraries.
Do not assume the same waiver is valid elsewhere. Record imported module paths,
actual CUDA/cuDNN/cuBLAS libraries, and package metadata on your own cluster.
FA4 installation is not evidence that it was selected: the measured decoder
CP2 THD and vision both selected **cuDNN FusedAttention1**.

### Commands

The wrapper contains no private host paths. Set paths **as visible inside the
container**. First inspect a no-GPU, no-Torch dry run:

```bash
export NNODES=4 GPUS_PER_NODE=4 TP=1 PP=2 CP=2 EP=8 GBS=64
export MASTER_ADDR=first-node MASTER_PORT=29600
export RESULTS_DIR=/shared/results/new-mock-cell
DRY_RUN=1 bash examples/multimodal_dev/benchmark_tflops/run.sh mock
```

Request resources using your own account, partition and container launcher.
For a Slurm allocation with the intended environment already visible:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  bash examples/multimodal_dev/benchmark_tflops/run.sh mock
```

One wrapper per node spawns GPUS_PER_NODE ranks. SLURM_NODEID supplies node rank;
NODE_RANK can override it for site launch integration. Add site-specific
container/mount flags to srun. Capture the complete, single training stream to
a log and preserve its exit code; if using tee, use `set -o pipefail`.
Use sbatch for reportable speed measurements and salloc for qualification.

For real data, copy and edit [blend3.example.yaml](blend3.example.yaml), including
train/val/test dataset paths, and set:

```bash
export ENERGON_PATH=/shared/datasets/blend3.yaml
export TOKENIZER_MODEL=/shared/models/Qwen3.5-35B-A3B
export RESULTS_DIR=/shared/results/new-real-cell
# Optional only when using a separately installed data dependency target:
# export ENERGON_OVERLAY=/shared/envs/energon-overlay
DRY_RUN=1 bash examples/multimodal_dev/benchmark_tflops/run.sh real
# Then, inside the approved allocation, use the same srun command with "real".
```

All paths must avoid whitespace due to the existing launcher's extra_args
splitting. It is still a GB200-oriented launcher: NVLINK_DOMAIN_SIZE defaults72
but may be explicitly set for your actual fabric. Do not spoof connectivity to
make HybridEP initialize. HybridEP topology compatibility and available memory
must be validated on the other cluster. A different dispatcher/backend or GPU
count is a separately labeled experiment, not demonstrated portability.

### Check and collect

The mock adapter comparison test requires the pinned reference source checkout:

```bash
git clone --filter=blob:none --no-checkout https://github.com/BestJuly/Megatron-LM.git /shared/reference-bestjuly
git -C /shared/reference-bestjuly checkout 5885d65a9c91d867d38c38cf00048014fd66bf04
export PR7_REFERENCE_SOURCE=/shared/reference-bestjuly
```

This is a test fixture dependency, not the training source. Set its path as
visible inside the test container. Without it, the reference adapter comparison
test is not runnable; do not count it as passed.

Before training, on allocated GPUs run the existing tests for geometry, native
Energon ownership, static THD, CP attention graphs and RoPE; the PR adds them
alongside the source. At minimum:

```bash
python -m pytest -q \
 tests/unit_tests/multimodal/test_reference_geometry_reporting.py \
 tests/unit_tests/multimodal/test_reference_mock_adapter.py \
 tests/unit_tests/multimodal/test_static_thd_metadata.py \
 tests/unit_tests/multimodal/test_energon_static_geometry.py \
 tests/unit_tests/multimodal/test_mdp_energon_prepartition.py
```

Then verify actual dataset materialization and full-model graph capture/finite
optimizer steps with your real topology. Unit tests do not establish E2E speed
or full-model gradient parity.

After a successful complete formal job, aggregate without importing Torch:

```bash
python examples/multimodal_dev/benchmark_tflops/collect_workload.py training.log \
  --world-size 16 --gbs 64 --start 4 --end 20 --total 20 > metrics-4-20.json
python examples/multimodal_dev/benchmark_tflops/collect_workload.py training.log \
  --world-size 16 --gbs 64 --start 10 --end 20 --total 20 > metrics-10-20.json
python examples/multimodal_dev/benchmark_tflops/workload_flops.py
python examples/multimodal_dev/benchmark_tflops/test_collect_workload.py
```

The calculator remains model-specific even when world/GBS change. Archive git
SHA/diff, full argv, world and parallel groups, data/tokenizer identities,
environment/backend selection, measurement windows, job status, raw log and JSON.
Separate mean/median and observed-rank memory from all-rank memory.
No credentials, real sample contents, or dataset shards belong in the PR.

## Observed evidence, not a promise of cross-cluster performance

Publication checks: all25 changed/new Python files parse without imports;
portable stdlib collector tests3PASS; formula reconciliation/additivity/units
PASS. Both wrapper dry runs reproduce the original effective training arguments
(apart from output paths). Replaying the original mock log with this collector
reproduces the reported mean/median values; see
[mock-418323-summary.json](mock-418323-summary.json).
These are packaging checks, not a GPU training run of the portable wrapper.

Mock formal job418323 on2026-09-14 completed20/20, exit0, original source seal
passed before/after. Window4–20 (17 samples):

| Metric | Mean | Median |
|---|---:|---:|
| Step ms | 4568.553 | 4417.400 |
| Native fixed16K decoder TFLOPs/GPU | 558.106 | 574.000 |
| Useful packed decoder TFLOPs/GPU | 292.110 | 300.515 |
| Useful vision TFLOPs/GPU | 16.333 | 16.469 |
| Useful combined TFLOPs/GPU | 308.443 | 317.379 |

The fixed decoder numerator is40.5670 PFLOPs/global step. Mean actual useful
decoder numerator is21.2339 PFLOPs/step. The fixed formula credits cross-document
attention that packed THD never performs; T is also only91.287% of64*16384.
This explains why558 and292 are simultaneously valid *different estimates*.
Component medians need not sum to the median total.
Observed max reserved memory114.492GiB covers logged ranks0/1/8/9 only.
Process-group cleanup warnings remained despite successful exit.

At publication preparation, real allocation418333 is pending. Existing related
tests27PASS and per-source actual loader/decode proof passed for all three
datasets. Capacity129 tests were added/reviewed but not yet allocated-run.
**Real full-model20-step/graph qualification and real formal speed are pending.**
No real throughput or cross-cluster pass is claimed.

## Relation to the reference implementation

Reviewed reference:
[BestJuly lit/mdp_fast_pass](https://github.com/BestJuly/Megatron-LM/tree/5885d65a9c91d867d38c38cf00048014fd66bf04/examples/multimodal_dev/doc/mdp)
and [PR62](https://github.com/BestJuly/Megatron-LM/pull/62).
Reference recipe used a different Qwen3.5 model, GB300, topology, MTP and MDP
runtime. This snapshot intentionally retains PR7 MDP, retain/cap and recompute
choices; it does not claim exact reference parity or attribute gains to one
library. Runtime changes include static THD graph metadata, graph-safe RoPE,
vision attention padding for the native72-dim head, and reporting counters.
Keep these changes distinct from the FLOPs-accounting correction itself.
