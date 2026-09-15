# Ongoing qualification status

As of 2026-09-15 17:08 UTC. This is an incomplete work log, not a completed sweep or performance result. The fixed-input work below is distinct from historical variable-mock measurements.

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
