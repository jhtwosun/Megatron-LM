# Corrected accounting snapshot: fixed-image mock sweep

Snapshot frozen at 2026-09-16T03:25:45Z. Six completed formal cells have accepted source, runtime, input-geometry and verifier evidence. The existing campaign-wide historical catalog is preserved separately; historical corrected values without sufficient proof remain unavailable, never filled from legacy rates.

The rates below are **modeled native decoder FLOPs/GPU/s over the whole VLM step**. They exclude vision FLOPs and are not measured hardware utilization or encoder-only throughput. The exact sealed native formula uses actual padded vocabulary248448 and KV channels128.

Configuration: PR7/PR131 hybrid VLM (configured `model_arch=qwen3vl`), 16 GB300 GPUs, BF16, TP1/PP2/decoderCP2/DP4/EP8/ETP1, encoderCP1, MBS1/GBS64, sequence16384. Each packed bin contains four4096-token raw samples. Sequence parallel was requested but disabled at TP1. All runs have20 iterations, initial consumed samples0, evaluation0.

| Images per raw sample | Image size | Primary median step ms | Corrected decoder mean TFLOPs/GPU/s | Corrected decoder pooled TFLOPs/GPU/s |
|---:|---:|---:|---:|---:|
|1|224×224|6475.1|243.9546|243.6629|
|1|448×448|6586.1|240.8134|240.4518|
|1|896×896|7245.8|217.2698|217.0908|
|2|448×448|6702.8|235.9466|235.6871|
|4|448×448|6955.6|225.9198|225.6909|
|8|448×448|7966.6|198.2781|198.2379|

Primary window4–20 includes17 samples. Supplemental10–20 includes11 samples and is recorded in the companion JSON. Mean per-step rates and pooled FLOPs/time are separately labeled. Peak memory is unavailable for these logs; a legacy verifier's zero sentinel is not a measurement.

Global decoder moments are Tpad=1,048,576 and Upad=4,294,967,296 per iteration for all six cells. Vision R/A differ by image count and size and are retained in JSON. Exact source-function, source-seal, corrected-receipt and canonical-verifier digests accompany each cell. Raw legacy logged TFLOPs are isolated in a clearly named JSON field, never used as corrected results. Its historical10–50 window label is stale for these20-step runs; actual available window is10–20.

## Sweep completion and unsuccessful qualifications

All six formal cells completed and passed their acceptance gates; the enclosing sweep completed successfully. The16-image448px and1-image1792px configurations failed interactive qualification with CUDA OOM. These are distinct qualification failures, not formal numeric results. The separate Mantis pair requires its own evaluator and parity gates; this snapshot does not supply its numerical results. Image count and size are explicit workload axes; decoder-only rates omit changing encoder work and do not establish a matched-workload TFLOPs winner. One run per cell does not establish run variance or a statistical winner.

## Data preparation scope

The preserved Mantis256 subset remains236 train/20 validation rows; it is not the full Mantis source. PixMo's111 selected source files are verified, and separate32-row and512-row native fixtures preserved every included conversation and original image byte, with zero exclusions. These are bounded preparation proofs, not full-native/blend readiness or performance measurements. No full-corpus result is substituted by a successful small fixture.
