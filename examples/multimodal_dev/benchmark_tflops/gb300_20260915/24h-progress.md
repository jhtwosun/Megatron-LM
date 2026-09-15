# Ongoing qualification status

As of 2026-09-15 18:12 UTC. This is an incomplete work log, not a completed sweep or performance result. The fixed-input work below is distinct from historical variable-mock measurements.

| Attempt | Observed outcome | Evidence limit |
|---|---|---|
|753721 native loader qualification|Failed before model execution because the test passed a historical encoder-CP extension argument unsupported by the upstream constructor|A narrow test adapter was reviewed and tested; the runtime recipe was not changed|
|753812 eight-cell native CPU loader gate|All eight planned fixed-input configurations passed the bounded native loader check|CPU preparation only; not eight model-training passes|
|753812 model qualification: one image, 224×224|Twenty uniquely ordered iterations completed with finite loss and gradients, zero skipped/NaN iterations, no fatal log errors, and clean step exit|Interactive qualification, not formal throughput evidence|
|753812 model qualification: sixteen images, 448×448|CUDA out-of-memory before the first completed iteration; the training step failed|No timing result; no silent reduction of batch size, image work, or memory-related stack settings|
|One image, 1792×1792 model qualification|Not started|Further execution remained blocked at the execution-approval gate after the OOM; feasibility is unknown|

The successful low-work cell used 16 GB300 GPUs, TP1/PP2/decoder-CP2/EP8/ETP1, MBS1/GBS64 and a 16384-token budget. Each of 16 ranks supplied 24 sampled first-call attention-layer receipts with matching Q/KV logical and storage boundaries: four 4096-token segments and repeated terminal endpoints to a 33-entry static layout. The independent clean 20-step exit is required alongside these pre-backend receipts; the samples do not prove every later call or historical execution.

The low-cell generated-input audit checked 320 records per rank, ordered sampler indices, four replicated CP/PP copies per logical DP stream, and matching tensor/descriptor hashes. This establishes the scoped qualification evidence, not historical replay identity or a corrected historical TFLOPs rate. The qualification allocation was intentionally released after terminal steps; no jobs were left occupying resources for this attempt.

No formal measurement for this new eight-cell sweep has been established. The OOM and unstarted cell remain visible rather than being omitted from a success-only catalog. Existing completed four-way profiles are separately summarized in [profile-summary.md](profile-summary.md); accounting replay and broader data-preparation work remain incomplete.

## Mantis dataloader preservation investigation

The new optimization candidate omits an unused JSON/base64 copy only when native planning already hands raw image descriptors to the owner-materializing iterator. It does not change image decoding, tokenization, packing, sample order, or model settings. Existing iterator CPU spans motivated investigation but do not establish a real-Mantis input critical path or predict a speedup.

Allocation 754294 completed its four-node environment checks and eight synthetic native descriptor/materializer tests. The first actual Mantis DP 0 save/restore test then failed while restoring the **original eager-JSON baseline**, before the candidate's real-loader capture or either model run. Pytest reported one failure and eight passes; the native step exited 1:0. The packing buffer attempted to restore a `None` key. Source inspection traced this to the encoder dropping restore metadata; it was not demonstrated to be a regression caused by JSON omission.

A separately attributed correctness patch now preserves the source restore key at preencoding and gives final packs the marker needed for Energon to compose all constituent keys. It is applied identically to **new corrected baseline and candidate** snapshots. These are not an unchanged original-PR131 baseline. Their sole optimization difference remains the JSON omission. Original failed snapshots and logs are preserved.

The corrected production diff, regression tests and source/data/tokenizer seals passed static review/checks. Fourteen focused native cases are authored, including all four logical DP streams, distinct-pack FIFO, exact tensor/descriptor/geometry fingerprints, four-owner pixels, native save/restore and the original missing-key mechanism. They remain **unexecuted**: the corrected native-only request was rejected at the execution-approval gate. Assertions were not waived. The allocation was intentionally released after terminal steps; corrected baseline/candidate model qualification and the formal pair have not run. Quiescent loader save/restore tests would not establish live-prefetch or training-checkpoint resume even if they passed.

| Artifact | SHA-256 |
|---|---|
|Original eager baseline task encoder|`a2909d5cecd5ecc0ae3a79a160aa1c160ff5c4623a287645c0bd0514fee74383`|
|Corrected eager baseline task encoder|`4f58edb316a26d716fe245838f08b57e2aaf40db997cdf884a7a179fdc6d612e`|
|Corrected JSON-omitting candidate task encoder|`4ae6f52e6cbfec673eb1a6dbeb1295e89a3211d4379879f3e5b5461b6cf8c247`|
|Corrected full native Mantis test|`c9d11f8229683585c467763505b988a93c48d60326b8ae882e33d66d07cdae99`|

These fingerprints identify retained private artifacts; neither their source changes nor raw manifests are bundled here. No corrected rate, new throughput measurement, or performance improvement is claimed. A future pair requires the same corrected baseline, workload, world size 16 / GBS 64 / MBS 1 and complete stack, with primary iterations 4–20 and supplemental iterations 10–20. Historical run 752807 cannot substitute for that matched baseline.
