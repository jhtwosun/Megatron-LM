# Ongoing qualification status

As of 2026-09-16 03:10 UTC. This is an incomplete work log, not a completed sweep. The fixed-input work below is distinct from historical variable-mock measurements. Explicit user exception approval reopened the reviewed execution gates; earlier denials remain historical facts, not current blanket prohibitions.

| Attempt | Observed outcome | Evidence limit |
|---|---|---|
|753721 native loader qualification|Failed before model execution because the test passed a historical encoder-CP extension argument unsupported by the upstream constructor|A narrow test adapter was reviewed and tested; the runtime recipe was not changed|
|753812 eight-cell native CPU loader gate|All eight planned fixed-input configurations passed the bounded native loader check|CPU preparation only; not eight model-training passes|
|753812 model qualification: one image, 224×224|Twenty uniquely ordered iterations completed with finite loss and gradients, zero skipped/NaN iterations, no fatal log errors, and clean step exit|Interactive qualification, not formal throughput evidence|
|753812 model qualification: sixteen images, 448×448|CUDA out-of-memory before the first completed iteration; the training step failed|No timing result; no silent reduction of batch size, image work, or memory-related stack settings|
|758212 model qualification: one image, 1792×1792|CUDA out-of-memory before a successful iteration; step .10 failed 1:0|Separate unchanged-stack high-side qualification; no formal timing result or silent workload reduction|

The successful low-work cell used 16 GB300 GPUs, TP1/PP2/decoder-CP2/EP8/ETP1, MBS1/GBS64 and a 16384-token budget. Each of 16 ranks supplied 24 sampled first-call attention-layer receipts with matching Q/KV logical and storage boundaries: four 4096-token segments and repeated terminal endpoints to a 33-entry static layout. The independent clean 20-step exit is required alongside these pre-backend receipts; the samples do not prove every later call or historical execution.

The low-cell generated-input audit checked 320 records per rank, ordered sampler indices, four replicated CP/PP copies per logical DP stream, and matching tensor/descriptor hashes. This establishes the scoped qualification evidence, not historical replay identity or a corrected historical TFLOPs rate. The qualification allocation was intentionally released after terminal steps; no jobs were left occupying resources for this attempt.

Formal job 758245 is running six lower-work cells in sequence: 1×224, 1×448, 1×896, 2×448, 4×448 and 8×448. The known 16×448 and 1×1792 OOM configurations are excluded, not relabelled successful. At this snapshot the first four cells completed their 20 training iterations; later cells remain in progress. Only separately accepted evaluator outputs may supply numerical results. Existing completed four-way profiles are separately summarized in [profile-summary.md](profile-summary.md); historical accounting replay and broader data preparation remain incomplete.

## Mantis dataloader preservation investigation

The new optimization candidate omits an unused JSON/base64 copy only when native planning already hands raw image descriptors to the owner-materializing iterator. It does not change image decoding, tokenization, packing, sample order, or model settings. Existing iterator CPU spans motivated investigation but do not establish a real-Mantis input critical path or predict a speedup.

Allocation 754294 completed its four-node environment checks and eight synthetic native descriptor/materializer tests. The first actual Mantis DP 0 save/restore test then failed while restoring the **original eager-JSON baseline**, before the candidate's real-loader capture or either model run. Pytest reported one failure and eight passes; the native step exited 1:0. The packing buffer attempted to restore a `None` key. Source inspection traced this to the encoder dropping restore metadata; it was not demonstrated to be a regression caused by JSON omission.

A separately attributed correctness patch now preserves the source restore key at preencoding and gives final packs the marker needed for Energon to compose all constituent keys. It is applied identically to **new corrected baseline and candidate** snapshots. These are not an unchanged original-PR131 baseline. Their sole optimization difference remains the JSON omission. Original failed snapshots and logs are preserved.

The first corrected native attempt in allocation 758212 passed ten tests, then failed a test assertion that required a tuple where the actual loader returned a list. The test-only v5 correction accepts both documented container forms when locating constituent keys; exact typed payload fingerprints and FIFO/restore assertions were not weakened. The corrected production sources were unchanged.

V5 then passed **36 native tests**, including 15 focused tests and nearby regressions: all four logical DP streams, distinct-pack FIFO, exact tensor/descriptor/geometry fingerprints, four-owner pixels and native save-two/restore-two behavior. Both corrected baseline and candidate subsequently completed ten ordered finite training iterations with zero skipped/NaN iterations, evaluation disabled, no fatal errors in all node logs, clean steps .7/.8 and passing post-run source/data/tokenizer seals. These are interactive qualification results, not throughput measurements. Quiescent loader save/restore does not establish live-prefetch or training-checkpoint resume.

Formal paired job **758425 is submitted**: corrected baseline then candidate, twenty iterations each, on the same sixteen GPUs with evaluation disabled. No numerical result is claimed while it is pending or running. The original standalone 256-record Mantis dataset remains separately preserved.

| Artifact | SHA-256 |
|---|---|
|Original eager baseline task encoder|`a2909d5cecd5ecc0ae3a79a160aa1c160ff5c4623a287645c0bd0514fee74383`|
|Corrected eager baseline task encoder|`4f58edb316a26d716fe245838f08b57e2aaf40db997cdf884a7a179fdc6d612e`|
|Corrected JSON-omitting candidate task encoder|`4ae6f52e6cbfec673eb1a6dbeb1295e89a3211d4379879f3e5b5461b6cf8c247`|
|V5 full native Mantis test|`f16316ace2ffea3baf728c05a5e6743aaa7fedcac1d44dc1f41d2810cb3d0162`|

These fingerprints identify retained private artifacts; neither their source changes nor raw manifests are bundled here. No corrected rate, new throughput measurement, or performance improvement is claimed for this dataloader comparison. A completed pair requires the same corrected baseline, workload, world size 16 / GBS 64 / MBS 1 and complete stack, with primary iterations 4–20 and supplemental iterations 10–20. Historical run 752807 cannot substitute for that matched baseline.

## Data deadline and preserved fallback

The data-readiness deadline was 2026-09-15 21:32:06 UTC. All owned download handles are now terminal. The remaining transfers ended on their configured deadline; shell exit zero records receipt completion, not successful completion of every file. Partial downloads were preserved, not deleted or silently accepted.

| Pinned source selection | Verified files | Remaining failed file / preserved bytes |
|---|---:|---|
|Mantis-Instruct|35 / 36|llava_665k_multi/train_images.zip / 42,547,065,657 bytes|
|M4-Instruct|40 / 41|TQA.zip / 8,237,613,056 bytes|
|PixMo|111 / 111|None in the selected manifest|
|ShareGPTVideo backing frames|17 / 17|None: 16 frame TAR archives plus README, not 17 media archives|

Receipt hashes, exact manifest membership, sizes and recorded verified LFS digests were independently checked. The deadline review did not rehash hundreds of gigabytes. Transfer completion alone does not establish native eligibility: whole-source preparation and the requested 1:1:1 native blend remain unmet, and M4 frame grouping, order, temporal semantics and annotation mapping remain unqualified.

After explicit exception approval, bounded PixMo preparation pilots of 32 and 512 source rows completed successfully in allocation 758212. Their input seals and output receipts were checked. These are preparation fixtures, not whole-source conversion, packed-loader/model qualification or training results. They do not modify the protected Mantis slice or prove that full preparation fits the remaining time budget.

The fallback is the **existing, separately preserved 256-record Mantis slice**, comprising 236 training and 20 validation records from llava_665k_multi, with evaluation disabled (`eval_iters=0`). It has not been overwritten, replaced by whole Mantis, or relabelled as a full-source blend. Prior qualification 752786 and formal run 752807 apply to that same standalone slice; this deadline decision is not a new formal run or performance result. Its selected ZIP members have CRC/local-SHA checks, not an authenticated whole-source ZIP hash. The demonstrated original loader restore failure also prevents a blanket checkpoint-resume claim.

The retained deadline receipt has SHA-256 `70de1bdabe4238bce5d4654e05f87bca8cf7ad30f80ae82e98df0dbff18a1ef9`; the slice preparation receipt has SHA-256 `fdfa1e70b139be8f53d4bbff421600114fcd2e0a7c0f8c13e2d5b642bbafccbe`. These identify evidence without exposing private paths. Publishing this record does not itself dispatch additional work or restart downloads.
