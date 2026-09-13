# Experimental integrated Dynamic CP: portable proxy

This is an **aggregate experimental branch**, not a new independent implementation
or a cleaned-up replacement for the existing stacked PRs. It integrates the current
D3/D4 runtime and required registered model adapters from c04357d34a6794406bae66986393bd1075f09d2e
with dev_mdp e10d8b7a234d423cb7a82b3ec2b6f81974a96871. The upstream greedy-packing and
397b_a17b_light additions are retained. Related existing work includes jhtwosun
PRs #43, #83, #90, #127 and #128 and BestJuly encoder/decoder CP PRs #61/#60.
The broad diff includes their dependencies and tests; do not review it as an
isolated gradient-order fix.

## What was executed, and what was not

On the pre-merge execution tree, the 8-GPU synthetic proxy baseline completed
50 iterations; the joint encoder/decoder Dynamic CP smoke completed two optimizer
steps with finite loss/gradients and no skipped/NaN iterations. Seven item-order
regressions and 65 existing gradient tests passed after correcting a stale test
publication tuple. The baseline median over iterations10–50 was9787.9ms.
Static and joint50-iteration measurements were still pending at preparation time:
**no speedup claim, full-window gradient parity, or merged-tree runtime PASS**.

The gradient fix slices gradients in original decoder replay order, binds views
to immutable item IDs, validates membership, and emits manifest/ledger order.
It never rearranges gradient rows. The proxy-only rotary forwarding seam keeps
head128 with rotary64; the ordinary model default is unchanged. A separate native
causal-THD adapter fix and external TE precision patch are required for the tested
stack. These are included explicitly rather than hidden in cluster mounts.

Merge conflict resolutions and the portable launcher are syntax-checked only.
The merged branch needs allocated validation on the destination cluster. No
checkpoints, data, credentials, private container image, Slurm allocation scripts,
or production-campaign artifacts are bundled.

## Recipe and supported scope

Three arms, identical world8/GBS128/MBS1/seq2048, TP1/PP1/EP1, decoderCPmax4:
baseline MDPoff; static MDP/LPT/vision packing/ECP4; joint the same features with
encoder and decoder Dynamic CP. The Qwen3.5-VL proxy has4decoder layers and1vision
layer, KV128/rotary.5 (64dimensions), BF16, Fused attention, per-layer full encoder
recompute, and **decoder recompute off**. This is not the full PR7 model.

All arms use the deterministic64-scenario mdp_mock pool and single sampler.
MDP arms use SOURCE_PIXEL_SIDECAR: producer-lazy/catalog materialization is **not**
measured. Static57600 is an inactive total chunk bound; joint26000 is a per-rank
degree-selection capacity, not equivalent flag units. Observed first plans use
encoderCP4/2 across domains and decoderCP4/2 across assignments.

The first-warmup observer checks the actual native domain source owner (64samples)
and nonowners(0), plus first-plan choices; baseline/static each capture64samples.
Hooks detach during warmup. These checks establish first-window identity only,
not the sample identities or degree distribution of every measured iteration.

Greedy token-budget packing plus dynamic encoder CP is explicitly rejected:
dynamic capture bypasses the greedy sample stream. Existing greedy static support
is retained; the merged combination is not silently advertised as supported.
Other topologies and arbitrary TE/Flash versions are unqualified.

## Environment

Use an allocated GPU environment, not a login node, with the normal Megatron
training dependencies. Reference execution used PyTorch2.11.0a0/CUDA13.1,
Transformer Engine2.12.0+5671fd36 (commit5671fd3675906cda1ade26c24a65d3dedd88eb89),
FlashAttention2.7.4.post1+nv26.2.44259020, cuDNN9.19.1(sm100), and genuine
nvidia-resiliency-ext>=0.6.0. Install the real package and dependencies; do not
rewrite version strings or provide substitute modules. The recorded NVRX0.6.0
stack used grpcio/grpcio-tools1.76.0 and protobuf6.31.1.

External TE patch:
`examples/multimodal_dev/patches/transformer_engine_5671fd36_thd_bf16_cp.patch`.
Only apply it to a private environment whose target file SHA256 is
`4e6292109a3bb1b589d71eed67377230578219ed73d2ae080a3b816431876104`.
The expected patched SHA256 is
`5f16ad06f60fd8bec3d21b56e07dcf027617528a0b195eb523a9982f212b6430`.
Verify the base hash, run `patch --dry-run -p1` from the directory containing
the `transformer_engine/` package, apply only after dry-run succeeds, and verify
the patched hash. Do not apply blindly to another release. The launcher requires
the patched hash before real execution. The patch retains upstream licensing.

## Run on two nodes with four GPUs each

Use the same checkout and environment on both nodes. Set NODE_RANK=0 on the first
node and NODE_RANK=1 on the second; use a reachable common rendezvous hostname and port.
The launcher starts native torchrun; resource allocation is deliberately external.

```bash
export NNODES=2 GPUS_PER_NODE=4 MASTER_ADDR="first-node-host" MASTER_PORT=29500
export NODE_RANK=0  # Set 1 on the second node.
python examples/multimodal_dev/scripts/run_dynamic_cp_proxy.py \
  --arm joint --train-iters 2 --output-dir /your/results/joint-smoke
```

Use `--dry-run` to inspect resolved argv without Torch imports, then run each
two-step smoke before full measurements. For each arm baseline/static/joint,
set `--train-iters 50` with a distinct output directory and capture stdout/stderr
using the destination scheduler. Run sequentially. Compare only within the same
world/GBS/environment. Require50unique steps, finite loss/gradients, zero skipped/
NaN iterations, clean exit, and measured iterations10–50. Do not interpret absent
memory/real-token telemetry as measuredzero; padded capacity isGBS*seq/step_time.

Allocated test commands:
```bash
python -m pytest tests/unit_tests/mdp/test_dynamic_cp_gradient_item_order.py -q
python -m pytest tests/unit_tests/mdp/test_dynamic_cp_d4_encoder_gradient.py -q
python -m pytest tests/unit_tests/mdp/test_config.py -q
python -m pytest tests/unit_tests/transformer/test_causal_cp_attention_kwargs.py -q
```
