# Corrected accounting snapshot: fixed-image mock sweep

Scheduled global tok/s is `GBS × configured sequence length × 1000 / median step ms`, rounded to one decimal. It is capacity-normalized at the reported median, not actual useful-content throughput or a mean of stepwise rates. No GPU-count multiplier is used. Failed/partial or missing-input rows are N/A. Encoder and encoder + decoder TF columns remain N/A unless separately source-bound and reviewed; legacy TF is never substituted. All TF components use a whole-VLM-step denominator rather than encoder-only elapsed time.
Per-GPU rates appear in parentheses only when world size 16 is independently validated; historical configuration-only world values yield N/A. TF/GPU is shorthand for TFLOPs/GPU/s. Paired rates follow the named baseline/candidate or CP1/CP2 order, irrespective of execution order.

The original decoder snapshot was frozen at 2026-09-16T03:25:45Z. Scheduled token rates and separately reviewed encoder/mixed-total estimates were added afterward from those same completed runs; no training or historical timing was replaced. Six completed formal cells have accepted source, runtime, input-geometry and verifier evidence. The existing historical catalog remains separate, with unavailable corrections never filled from legacy rates.

The decoder column is the accepted **attended-padded native decoder model**, using actual padded vocabulary248448 and KV channels128. The encoder column is a **useful-content matmul model** that excludes real attention head-width padding from72 to128 and non-matmul operations. Their sum is a mixed modeled total, not a uniformly executed or hardware-counter FLOPs rate. All components use the same whole-VLM-step/world denominator, not encoder-only elapsed time.

Configuration: PR7/PR131 hybrid VLM (configured `model_arch=qwen3vl`), 16 GB300 GPUs, BF16, TP1/PP2/decoderCP2/DP4/EP8/ETP1, encoderCP1, MBS1/GBS64, sequence16384. Each packed bin contains four4096-token raw samples. Sequence parallel was requested but disabled at TP1. All runs have20 iterations, initial consumed samples0, evaluation0.

| Images per raw sample | Image size | Primary median step ms | Scheduled tok/s global (per GPU) | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean | Corrected decoder pooled TFLOPs/GPU/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 224×224 | 6475.1 | 161939.7 (10121.2) | 1.2528 | 243.9546 | 245.2073 | 243.6629 |
| 1 | 448×448 | 6586.1 | 159210.5 (9950.7) | 5.3647 | 240.8134 | 246.1781 | 240.4518 |
| 1 | 896×896 | 7245.8 | 144715.0 (9044.7) | 25.3971 | 217.2698 | 242.6669 | 217.0908 |
| 2 | 448×448 | 6702.8 | 156438.5 (9777.4) | 10.5125 | 235.9466 | 246.4591 | 235.6871 |
| 4 | 448×448 | 6955.6 | 150752.8 (9422.0) | 20.1315 | 225.9198 | 246.0513 | 225.6909 |
| 8 | 448×448 | 7966.6 | 131621.5 (8226.3) | 35.3368 | 198.2781 | 233.6148 | 198.2379 |

Primary window4–20 includes17 samples. Supplemental10–20 includes11 samples and is recorded in the companion JSON. Mean per-step rates and pooled FLOPs/time are separately labeled. Peak memory is unavailable for these logs; a legacy verifier's zero sentinel is not a measurement.

Global decoder moments are Tpad=1,048,576 and Upad=4,294,967,296 per iteration for all six cells. Vision R/A differ by image count and size and are retained in JSON. Exact source-function, source-seal, corrected-receipt and canonical-verifier digests accompany each cell. Raw legacy logged TFLOPs are isolated in a clearly named JSON field, never used as corrected results. Its historical10–50 window label is stale for these20-step runs; actual available window is10–20.

## Reproducing the encoder and mixed total

Let `R=sum(t*h*w)` and `A=sum(t*(h*w)^2)` over the actual pre-merge patch grids. With vision hidden1152,27layers, nongated GELU FFN4304, patch input width1536,2×2merging and decoder hidden2048, global encoder FLOPs are:

```text
3 * [2*R*1536*1152
     + 27*(8*R*1152^2 + 4*A*1152 + 4*R*1152*4304)
     + 2*(R/4)*(4608^2 + 4608*2048)]
```

The terms count patch projection, QKV/output projections, bidirectional QK/AV attention, two FFN projections and one two-layer merger. FMA=2 and training=3×forward are accounting conventions, not an exact backward instruction count. Position interpolation, normalization, bias/GELU/RoPE/softmax, optimizer, communication/loading and72→128attention padding are excluded. Per step, divide FLOPs by `16 * step_ms * 1e9`; average the17primary stepwise rates. Add encoder and accepted padded-native decoder before rounding. Do not add the source helper's legacy decoder or legacy total.

Source binding: executed sweep seal SHA256 `9239e622e91bf9e21c726a598e13c1b1fabc02d8d4a48591eb5ddb7cf1e55ceb`; `examples/multimodal_dev/benchmark_tflops/workload_flops.py` SHA256 `ef44f08a872b083d2547325c954f09f4e97ae2fcfc309b66787b9e662ae9e4b7`; `models/qwen35_vl/vision_encoder.py` SHA256 `a57821b3804253741009fb9acaa7ec6976cca91fb82de8af403a0b22fe3ef5fd` (relative to `examples/multimodal_dev`). The source-bound component receipt SHA256 is `b2a6f61279dd06c5b9fdf2fa2315ab0ae629909325571ecc71e6fdd424d2746a`; it binds all seven relevant source files, accepted receipt hashes and each17/11window input/output row. Original public decoder JSON and canonical results remain unchanged.

## Sweep completion and unsuccessful qualifications

All six formal cells completed and passed their acceptance gates; the enclosing sweep completed successfully. The16-image448px and1-image1792px configurations failed interactive qualification with CUDA OOM. These are distinct qualification failures, not formal numeric results. The separate Mantis pair requires its own evaluator and parity gates; this snapshot does not supply its numerical results. Image count and size are explicit workload axes; decoder-only rates omit changing encoder work and do not establish a matched-workload TFLOPs winner. One run per cell does not establish run variance or a statistical winner.

## Data preparation scope

The preserved Mantis256 subset remains236 train/20 validation rows; it is not the full Mantis source. PixMo's111 selected source files are verified, and separate32-row and512-row native fixtures preserved every included conversation and original image byte, with zero exclusions. These are bounded preparation proofs, not full-native/blend readiness or performance measurements. No full-corpus result is substituted by a successful small fixture.
