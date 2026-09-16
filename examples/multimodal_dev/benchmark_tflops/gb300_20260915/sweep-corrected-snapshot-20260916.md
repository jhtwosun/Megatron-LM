# Corrected accounting snapshot: fixed-image mock sweep

Snapshot frozen at 2026-09-16T03:11:57Z. Four completed formal cells have accepted source, runtime, input-geometry and verifier evidence. The existing campaign-wide historical catalog is preserved separately; historical corrected values without sufficient proof remain unavailable, never filled from legacy rates.

The rates below are **modeled native decoder FLOPs/GPU/s over the whole VLM step**. They exclude vision FLOPs and are not measured hardware utilization or encoder-only throughput. The exact sealed native formula uses actual padded vocabulary248448 and KV channels128.

Configuration: PR7/PR131 hybrid VLM (configured `model_arch=qwen3vl`), 16 GB300 GPUs, BF16, TP1/PP2/decoder CP2/DP4/EP8/ETP1, encoder CP1, MBS1/GBS64, sequence length 16384. Resolved architecture has 48 decoder layers, 27 vision layers, decoder hidden size 2048 and 128 experts; the launcher label is not a claim of official model equivalence. Each packed bin contains four 4096-token raw samples. Sequence parallel was requested but disabled at TP1. All runs have 20 iterations, initial consumed samples 0 and evaluation 0. Job 758245 training steps 1, 3, 5 and 7 correspond to the table rows, respectively; exact experiment IDs are retained in the JSON.

| Images per raw sample | Image size | Primary median step ms | Corrected decoder mean TFLOPs/GPU/s | Corrected decoder pooled TFLOPs/GPU/s |
|---:|---:|---:|---:|---:|
|1|224×224|6475.1|243.9546|243.6629|
|1|448×448|6586.1|240.8134|240.4518|
|1|896×896|7245.8|217.2698|217.0908|
|2|448×448|6702.8|235.9466|235.6871|

Primary window4–20 includes17 samples. Supplemental10–20 includes11 samples and is recorded in the companion JSON. Mean per-step rates and pooled FLOPs/time are separately labeled. Peak memory is unavailable for these logs; a legacy verifier's zero sentinel is not a measurement.

Global decoder moments are Tpad=1,048,576 and Upad=4,294,967,296 per iteration for all four cells. Vision R/A differ by image count and size and are retained in JSON. Exact source-function, source-seal, corrected-receipt and canonical-verifier digests accompany each cell. Raw legacy logged TFLOPs are isolated in a clearly named JSON field, never used as corrected results. Its historical10–50 window label is stale for these20-step runs; actual available window is10–20.

## Pending and unsuccessful configurations

The4/8-image448px formal cells are not yet accepted in this snapshot; no provisional rates are shown. The16-image448px and1-image1792px configurations failed interactive qualification with CUDA OOM. These are distinct qualification failures, not formal numeric results. This snapshot does not imply that the enclosing sweep or ongoing Mantis pair has completed. One run per cell does not establish run variance or a statistical winner.

## Data preparation scope

The preserved Mantis256 subset remains236 train/20 validation rows; it is not the full Mantis source. PixMo's111 selected source files are verified, and separate32-row and512-row native fixtures preserved every included conversation and original image byte, with zero exclusions. These are bounded preparation proofs, not full-native/blend readiness or performance measurements. No full-corpus result is substituted by a successful small fixture.
