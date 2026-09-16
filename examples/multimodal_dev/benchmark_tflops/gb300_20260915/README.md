# GB300 experiment and accounting supplement — 2026-09-15

**Working draft: historical correction gaps remain explicit.** The 231-artifact historical catalog preserves observed timings and original legacy native TFLOPs estimates; its corrected rates remain `null` where executed formula, resolved arguments, consumed inputs and final attention boundaries lack sufficient proof. Newly accepted fixed-input sweep accounting is a separate dated snapshot, not a retroactive correction of those historical records. Null means unavailable, not zero or no work. Corrected native modeled decoder TFLOPs is not total VLM work or hardware-counter throughput.

This is a new current-thread supplement on PR131 head `bbba1001543a9c99a26ae8a8370b40e005c80d80`. It does not replace the separate historical ledger in [NATIVE_ACCOUNTING.md](../NATIVE_ACCOUNTING.md), and does not import that ledger's experiment inputs. Measured source bases and adaptations are distinct from this documentation branch.

## Reading guide

- [New accepted corrected-accounting snapshot](sweep-corrected-snapshot-20260916.md) and [machine-readable rates and proof digests](sweep-corrected-snapshot-20260916.json): six formal fixed-input cells, corrected native modeled decoder rates with primary/supplemental windows, separate OOM configurations and explicit accounting limits.

- [Entire owned-campaign catalog](campaign-wide-catalog.md): archived and active namespaces,231 result artifacts,351 experiment directories, duplicate/failure/unknown distinctions and correction gaps.
- [Per-experiment recorded details](per-experiment.md) and [machine-readable fields](per-experiment.json): all 231 artifacts, recorded stack/recipe fields, provisional attempt groups, fingerprints, and missing references. These are descriptive records, not 231 newly verified successes; five ambiguous world-size products are explicitly flagged.
- [Data distributions and configurations](data-and-configs.md): generator versus prepared-corpus versus consumed-geometry evidence, source/data pins, and known gaps.
- [Evidence fingerprints](evidence-fingerprints.md): retained result and raw-log SHA-256 values. Raw logs, media, and private filesystem paths are intentionally not bundled.
- [Completed four-way profile evidence](profile-summary.md): finalized packing/latest continuation, process-aware phase observations, and explicit capture/configuration limitations. Diagnostic only, not a speed ranking.
- [Scoped accounting utilities](accounting-utilities.md): seven standard-library tests and native formula equivalence checks; historical native replay remains unexecuted. Tool availability does not establish corrected rates or historical boundary identity.
- [Ongoing qualification status](24h-progress.md): completed fixed-input sweep and first Mantis pair, both high-work OOM outcomes, native/model qualification and bounded PixMo preparation. Pending probes have no numerical results.

## Results and analysis at a glance — 2026-09-16

Six fixed-input formal cells completed in job 758245; two larger workloads failed interactive qualification with CUDA OOM. Both corrected Mantis dataloader variants completed formal job 758425 and passed descriptive pair checks; no stable winner or corrected nonstatic rate is invented. The 231 historical result artifacts below remain an audit inventory, not 231 new successes or a combined total including these new cells.

### Fixed-input sweep: six accepted cells and two OOM qualifications

This is the PR7/PR131 hybrid VLM configured as `model_arch=qwen3vl`, not a claim of official model equivalence: 48 decoder layers, 27 vision layers, decoder hidden size 2048, 128 experts, padded vocabulary 248448 and KV channels 128. The common stack is 16 GB300 GPUs, BF16, TP1/PP2/decoder CP2/DP4/EP8/ETP1, encoder CP1, MBS1/GBS64, sequence length 16384, HybridEP (32 SMs, chunks 128), distributed optimizer, MDP fused-window retain with cap 131072, no CUDA graphs, no recomputation, no MTP and effective gradient/parameter overlap disabled. Sequence parallel is requested but effectively false at TP1. All formal cells use 20 iterations, LR warmup/decay 2/20, evaluation 0 and random initialization without checkpoint loading.

Images below are **per raw sample**. Four raw 4096-token documents fill each packed bin, so image count per bin is four times the table count. Static attention metadata has 32 segments (33 cumulative endpoints), with four real segments and repeated terminal endpoints. The primary window is iterations 4–20 (17 samples); supplemental 10–20 (11 samples) is separately preserved in JSON.

| Images/raw sample | Side length | Outcome | Primary median step ms | Corrected decoder TFLOPs/GPU/s mean | Pooled decoder TFLOPs/GPU/s |
|---:|---:|---|---:|---:|---:|
|1|224|Formal accepted|6475.1|243.9546|243.6629|
|1|448|Formal accepted|6586.1|240.8134|240.4518|
|1|896|Formal accepted|7245.8|217.2698|217.0908|
|2|448|Formal accepted|6702.8|235.9466|235.6871|
|4|448|Formal accepted|6955.6|225.9198|225.6909|
|8|448|Formal accepted|7966.6|198.2781|198.2379|
|16|448|Qualification OOM|unavailable|unavailable|unavailable|
|1|1792|Qualification OOM|unavailable|unavailable|unavailable|

These corrected numbers execute the sealed **native decoder model** using resolved arguments and verified input boundaries, divided by whole-VLM step time and 16 GPUs. They exclude changing vision FLOPs and are not hardware-counter throughput. Global decoder moments remain Tpad=1,048,576 and Upad=4,294,967,296 per step. As image work increases, step time rises while decoder-only rates fall; this does **not** prove that GPU compute utilization falls. Image count/size are workload axes, not equal-work optimization pairs. One run per cell does not establish variance or a statistical winner. Peak memory is unavailable; a legacy zero sentinel is not zero memory use.

### Earlier CP, image-area and four-way experiments

The controlled encoder-CP pair is job 749724: its supplemental 10–20 step medians are 4358.8 ms (CP1) and 4538.4 ms (CP2). Separate image-area pairs at 1×/2×/4×/8× used jobs 750008/750020/750022/750023, with their own paired allocations and logged memory coverage. Those area multipliers are **not** the new side-length/image-count sweep above. Their retained supplemental medians are listed below; primary and supplemental windows must not be mixed into one percentage claim. At fixed 16 GPUs, increasing encoder CP also changes image ownership across the fixed inner-DP group, so it does not simply halve all per-rank vision work or memory. Corrected historical rates remain unavailable until historical input/backend boundaries are established.

Within the single four-way measurement job 752159, native variable-mock median steps were 13804.5 ms (ordinary PR7), 11479.4 ms (nonfused MDP PR7), 9820.5 ms (fused PR7) and 8901.4 ms (fused PR131), all in iterations 10–50. These are observed whole-step timings with declared mode/source axes, not encoder-only savings. Exact historical consumed geometry is unarchived, and PR-specific normalization environment differences remain disclosed. Standalone baseline 751915 must not be substituted into this set. No repeated/randomized variance estimate or categorical winner is claimed.

### Profiling: what is known and what is not

Four completed profile cells have integrity/stability and 16-worker coverage. Baseline/nonfused profiles came from job 752159; fused PR7/latest continuations came from 753568/753569. The latter use LR/evaluation 2/20/0 rather than 5/50/default and different racks. Capture 5–8 maps to displayed 6–8 only by source inference, not explicit iteration anchors.

| Profile cell | Worker kernel span s | Selected vision ranges/node | Forward-bridge kernel sums, nodes 0–3, s |
|---|---:|---|---|
|Ordinary PR7|64.43–64.60|Named outer range absent; vision executes|not applicable|
|Nonfused MDP PR7|66.83–66.98|192|2.870 / 20.881 / 9.467 / 5.976|
|Fused PR7|40.10–40.23|48|0.319 / 0.466 / 1.785 / 0.715|
|Fused PR131|47.98–48.10|48|0.738 / 0.885 / 2.476 / 4.096|

Process-aware CUDA API/kernel correlation matched all 4,220,973 packing kernels and 4,277,998 latest kernels. HybridEP/NCCL and synchronization events show substantial tails, but overlapping event sums cannot become wall-time fractions, global utilization or cross-node critical paths. A waiting rank is not necessarily the cause. The 192→48 count describes fusion granularity, not four times less image work. Interpolation-associated GPU sums are small compared with their enclosing CPU ranges; remaining host time includes unattributed work/profiler overhead. Thus traces motivate scoped hypotheses, not a proven pacing rank or automatic cache/kernel rewrite.

### Real Mantis, loader optimization and data readiness

The original 256-record Mantis slice remains separately protected (236 train, 20 validation; evaluation disabled), not replaced by whole Mantis or a blend. Historical formal run 752807 has a 7221.1 ms primary 4–20 median; real conversations/images differ from mock, so this is not a matched-work speedup comparison. Its corrected nonstatic decoder rate remains unavailable.

The loader candidate removes an unused JSON/base64 image-descriptor copy only when raw descriptors are already handed off. The original eager baseline's missing restore-key bug was separately corrected in **both** new variants. A test-only tuple/list ownership correction preserved exact payload/FIFO checks. V5 passed 36 native tests, including all four DP streams and save-two/restore-two/four-owner pixel checks; both variants passed clean ten-step model qualification. Formal job 758425 then completed and passed individual verification for baseline and candidate twenty-step runs on the same 16 GPUs. All logged argument keys/values match except output directory; all twenty logged T/U/R/A moments match. World size 16, GBS 64, MBS 1, full stack and measurement windows are identical.

| Mantis pair window | Samples | Corrected eager baseline median ms | JSON-omitting candidate median ms | Observed step reduction |
|---|---:|---:|---:|---:|
|Primary 4–20|17|7498.3|7306.2|2.5619%|
|Supplemental 10–20|11|7318.8|7306.2|0.1722%|

These are descriptive observations from **one ordered pair**, not a stable winner or causal speedup. The window sensitivity is visible; run-to-run variance is unknown. All-step logged moments and bounded native input parity do not replace an archive of every consumed token/image. Corrected native rates remain null because final nonstatic attended boundaries are not archived; peak memory is unavailable. Historical run 752807 is not this pair's baseline. Quiescent loader restore is not proof of live-prefetch or training-checkpoint resume.

Whole-source transfer status is Mantis 35/36, M4 40/41, PixMo 111/111 and backing frames 17/17 **files** (16 TAR archives plus README). Failed partial downloads are preserved. PixMo 32-row, 512-row and 4096-row native fixtures passed; the last included all 4096 rows with zero exclusions and verified output hashes/image-conversation parity. These are not full-corpus preparation or training. An approximate 118–132 minute full-preparation estimate exceeds the remaining autonomy budget, so whole preparation is deferred rather than claimed complete. The estimate includes fixture/import effects and is not a guaranteed steady-state bound. M4 temporal/frame mapping remains unqualified; the requested whole-native 1:1:1 blend is unmet. Missing data or denied/failed attempts are not silently replaced by a successful small slice.

### Completion limits and next gates

| Requirement area | Established | Remaining / unavailable |
|---|---|---|
|Campaign publication/accounting|231 historical artifacts retained; six new formal corrected decoder-model cells|Historical correction remains unavailable without original boundary/identity proof|
|Fixed-image workload sweep|Six lower cells accepted; both high-work OOM outcomes retained|No formal timing for failed shapes; no variance-based winner|
|Profiling and analysis|Four finalized four-way profiles; process-aware attribution and explicit caveats|Global pacing/critical path and unprofiled causal validation remain unknown|
|Encoder CP optimization|Historical controlled CP pairs and diagnostic profiles|New optimized encoder CP versus CP-off achieving at least 5% across three independent pairs was not performed|
|Whole real-data preparation/blend|Verified selected transfers, preserved Mantis256, bounded PixMo fixtures through 4096 rows|Whole-native Mantis/M4/PixMo blend and full temporal semantics unmet; full preparation deferred beyond remaining budget|
|Reproducibility utilities/replay|Guarded published utilities, tests and pinned replay plan|Historical replay/backend/input-identity proof not established by this snapshot|
|Dataloader correctness/efficiency|Shared restore repair, native/model gates, accepted descriptive formal pair|Run variance/causal improvement unproven; no corrected nonstatic rate or broad resume claim|

The campaign is not fully complete. All unstarted, failed, partial and unavailable outcomes remain part of the record. Supporting documents retain exact hashes, schemas and per-window details; this README is the consolidated results/analysis entry point.

## Historical timing audit

Each row below comes from an accepted legacy result artifact. Step time is the median in the stated window; native TFLOPs is the legacy verifier arithmetic mean, not a hardware measurement or median. The old raw label `iters 10-50` sometimes describes only the filter: a20-step run has actual10–20 coverage (11 samples), while10–50 has41 samples. These are separate experiments unless explicitly paired.

| Result identifier | Actual window | GBS | Step median ms | Legacy native TFLOPs/GPU mean | Corrected decoder TFLOPs/GPU |
|---|---|---:|---:|---:|---|
| EXP-PR131-ENCODER-CP1-GB300 | 10–20 | 64 | 4170.2 | 601.9 | null |
| EXP-PR131-ENCODER-CP1-PAIR749724-GB300 | 10–20 | 64 | 4358.8 | 578.1364 | null |
| EXP-PR131-ENCODER-CP2-PAIR749724-GB300 | 10–20 | 64 | 4538.4 | 551.1273 | null |
| EXP-PR131-IMAGE1X-CP1-GB300 | 10–20 | 64 | 4144.3 | 603.5182 | null |
| EXP-PR131-IMAGE1X-CP2-GB300 | 10–20 | 64 | 4293.5 | 583.2 | null |
| EXP-PR131-IMAGE2X-CP1-GB300 | 10–20 | 64 | 4151.5 | 594.5636 | null |
| EXP-PR131-IMAGE2X-CP2-GB300 | 10–20 | 64 | 4267.2 | 578.5545 | null |
| EXP-PR131-IMAGE4X-CP1-GB300 | 10–20 | 64 | 4574.4 | 542.6 | null |
| EXP-PR131-IMAGE4X-CP2-GB300 | 10–20 | 64 | 4806.8 | 512.7182 | null |
| EXP-PR131-IMAGE8X-CP1-GB300 | 10–20 | 64 | 5571.4 | 452.8364 | null |
| EXP-PR131-IMAGE8X-CP2-GB300 | 10–20 | 64 | 6054.7 | 414.6909 | null |
| EXP-PR131-MANTIS16-752807-20-GB300-legacy-window | 10–20 | 64 | 7221.1 | 341.0182 | null |
| EXP-PR131-TFLOPS16-SEQ4096-GBS256 | 10–50 | 256 | 7296 | 217.3512 | null |
| EXP-PR7-FOURWAY-751915-pr7_baseline | 10–50 | 64 | 14961 | 162.1098 | null |
| EXP-PR7-FOURWAY-752159-pr131_latest | 10–50 | 64 | 8901.4 | 285.1317 | null |
| EXP-PR7-FOURWAY-752159-pr7_baseline | 10–50 | 64 | 13804.5 | 176.2098 | null |
| EXP-PR7-FOURWAY-752159-pr7_mdp | 10–50 | 64 | 11479.4 | 209.8732 | null |
| EXP-PR7-FOURWAY-752159-pr7_packing | 10–50 | 64 | 9820.5 | 241.1366 | null |

Mantis job752807 also has a primary4–20 window (17 samples): step median7221.1ms and legacy native TFLOPs mean342.55882352941177. Its10–20 row above is a supplemental representation of the same run, not another experiment. Corrected padded-attention rates remain null. Existing useful-content decoder/vision estimates are distinct model outputs and are not silently relabelled as corrected native rates.

All main cells use16GB300 GPUs, TP1/PP2/decoderCP2/EP8/ETP1, MBS1/GBS64/sequence16384, except the explicitly separate TF750157 cell: TP1/PP1/CP1, DP16, EP8/ETP1, MBS1/GBS256/sequence4096. That cell has one fixed448×448 image and ordinary vision execution with MDP off. It must not be compared as matched work to the CP sweep merely because some global content moments coincide.

## Pairing and interpretation

The controlled CP comparison is paired job749724, not standalone749255 spliced with a later CP2. Sweep jobs750008/750020/750022/750023 each contain CP1 and CP2 on the same physical allocation for1×/2×/4×/8× image area respectively. Each is one ordered pair, without a repeated/randomized variance estimate. Across sizes, vision workload and some placements differ. No statistical winner or encoder-only latency follows.

The four-way comparison belongs wholly to752159: ordinary PR7, nonfused MDP PR7, fused PR7, fused PR131. Completed baseline751915 is standalone historical evidence and is not substituted into that set. Original native mock runs lack per-step consumed geometry, so deterministic recipe equality is not proof of historical input identity. PR-specific normalization environment differences must be declared alongside the intended mode/source axes.

Mantis uses real variable conversations and different actual work; its step time is not a matched-work speedup over native mock. The small prepared corpus repeats during training. A global batch counts packs, not conversations. Neither legacy native TFLOPs nor useful-content modeled rates establish hardware instruction throughput, utilization, or encoder-only performance.

Logged memory coverage is incomplete: sweep ranks0,1,8,9; TF750157 rank0 only; Mantis supplemental coverage ranks0,1,8,9. Legacy zero memory fields in four-way result JSONs are missing extraction, not zero GPU memory. No all-rank peak-memory Pareto claim is made.

## Correction readiness

| Family | Retained evidence | Still required before publishing a corrected numerical rate |
|---|---|---|
|Historical image-area sweep and TF750157|Logged content moments, fixed-grid source, original per-step timings|Native replay of all consumed DP streams; final logical and physical cumulative sequence boundaries; valid-attention semantics; source/argument identity|
|Variable reference CP pair|All20 rounded content geometry rows; deterministic reference recipe|Actual sampler/window replay matching original geometry and final attention boundaries, token/descriptor hashes|
|Native four-way|Executed source/args, deterministic generator, accepted timings|Historical consumed identity is not archived; reconstruction must disclose that limitation and cannot borrow other-run geometry|
|Mantis|Pinned prepared bytes/tokenizer/template, all20 content geometry rows, actual Energon configuration|All-DP native replay matching geometry and iterator/RNG/skip semantics; final attention metadata and ordered input hashes|

Content T/U does not generally equal final attended padded T/U. Physical storage offsets also need not mean attended padded length: nonstatic paths preserve logical cumulative lengths while changing physical offsets. Repeated endpoints, dummy tails, segment alignment, and TE interpretation must be established on the actual executed path. No arbitrary correction factor or inference from full-bin aggregate content alone is accepted.

The PR7/PR131 executed native FLOP function has identical extracted-source SHA-256 `d900fc09ea21b6aa42e3879cb47f729650c64e6076116faf86dd34b7e3dabd8e`, while complete training files differ. Formula equality is not input-geometry equality. Resolved vocabulary is248448, not the early argument248320 or preinitialization None. Mean-of-stepwise rates, median rates, and pooled work/time must remain separately named.

## Profiles and failures

| Job / evidence | Outcome | Permitted use |
|---|---|---|
|750305 fixed8× CP1/CP2 profiles|Both20-step producers completed; four finalized SQLite files and16 workers per cell|Diagnostic summaries only; configured capture5–8 implies displayed6–8 from source, not explicit iteration NVTX anchors|
|752159 baseline and nonfused MDP profiles|Both20-step producers completed; integrity/stability/16-worker gates passed|Reviewed process-aware summaries; no global wall-time fractions or pacing-rank proof|
|752159 packing/latest profiles|Outer2-hour timeout; packing before iteration1, no finalized trace; latest not started|No result or comparison|
|753568 packing /753569 latest continuation|Both20-step producers completed0:0; eight finalized SQLite files passed integrity/stability,16 workers per cell|[Reviewed diagnostic summaries](profile-summary.md); LR/evaluation/rack differences disclosed, no percentage speedup or global pacing claim|
|751915 MDP|Stopped after iteration8 with collective watchdog; human-authorized cleanup/retry|Failure diagnosis only; its completed baseline remains standalone|
|749264|Intentional pending-job supersession|No measurement|
|749875|Missing NVTX helper import|Failed profile attempt, no timing result|
|750661|Interpreter-wrapper failure before iteration1|No measurement|
|751517|Ordinary grid-shape failure before iteration1|No measurement|
|752462 Mantis qualification|Ten finite training iterations, then validation-sidecar failure|Training execution evidence only, not clean qualification or speed result|
|752786 Mantis qualification|Ten steps, eval0, finite, clean exit|Readiness only|
|752807 Mantis formal|Twenty steps, eval0, finite, clean exit|Training-only measurement, no validation/convergence claim|
|Nemotron preparation|All-row CPU token gate and bounded CPU pack/pixel checks only|No model execution or performance result|

For750305, vision-forward CP2 has additional NCCL activity; summed launches and CPU phases overlap and cannot apportion a whole-step timing difference. For752159 MDP, forward bridge all-gather has direct process-aware launch attribution; large ReduceScatter totals do not establish backward-bridge ownership without a corresponding range. Waiting-heavy ranks need not cause the delay. CPU API duration, GPU event sums, and NVTX spans must not be added as disjoint costs. Traces do not explain all startup/profiler overhead.

## Publication boundaries

This package is a transparent draft, not a completed campaign-wide accounting correction. No measured source, original accepted JSON, historical ledger, or raw log is overwritten. Historical corrected columns remain null pending proof and independent review; the separate new snapshot contains only accepted scoped decoder-model rates. Publishing these documents does not itself download checkpoints or dispatch additional training.

## Complete historical per-experiment inventory

All 231 historical artifacts are listed below, including failed/partial records and alternate-window artifacts. They are not 231 successful or newly corrected runs. Corrected historical TFLOPs is unavailable; original legacy rates remain in the unchanged [machine-readable audit](per-experiment.json). Five topology products of 128/256 are unvalidated and must not normalize rates. The six new accepted cells above are additional, not replacements.

<details>
<summary>Show all 231 historical artifact records</summary>

| Artifact | Namespace / experiment | Model | World / GBS / seq | Window | Status | Corrected decoder TF / step ms | Missing references |
|---|---|---|---|---|---|---|---|
| artifact-001 | qwen3-phase4-final / EXP-040 | qwen3_30b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 4122.6 | source_pin, dataset, owned_original_log |
| artifact-002 | qwen3-phase4-final / EXP-041 | qwen3_30b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 10899.2 | source_pin, dataset, owned_original_log |
| artifact-003 | qwen3-phase4-final / EXP-042 | qwen3_30b_a3b | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 4602.4 | source_pin, dataset, owned_original_log |
| artifact-004 | qwen3-phase4-final / EXP-043 | qwen3_30b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 8023.3 | source_pin, dataset, owned_original_log |
| artifact-005 | qwen3-phase4-final / EXP-045 | qwen3_30b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5048.1 | source_pin, dataset, owned_original_log |
| artifact-006 | qwen35vl-phase0-3 / EXP-001 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7126.1 | source_pin, dataset, owned_original_log |
| artifact-007 | qwen35vl-phase0-3 / EXP-002 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7038.2 | source_pin, dataset, owned_original_log |
| artifact-008 | qwen35vl-phase0-3 / EXP-003 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7091.9 | source_pin, dataset, owned_original_log |
| artifact-009 | qwen35vl-phase0-3 / EXP-004 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7115 | source_pin, dataset, owned_original_log |
| artifact-010 | qwen35vl-phase0-3 / EXP-005 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7121.6 | source_pin, dataset, owned_original_log |
| artifact-011 | qwen35vl-phase0-3 / EXP-006 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7381.6 | source_pin, dataset, owned_original_log |
| artifact-012 | qwen35vl-phase0-3 / EXP-007 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7147.8 | source_pin, dataset, owned_original_log |
| artifact-013 | qwen35vl-phase0-3 / EXP-008 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7173.3 | source_pin, dataset, owned_original_log |
| artifact-014 | qwen35vl-phase0-3 / EXP-009 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7433 | source_pin, dataset, owned_original_log |
| artifact-015 | qwen35vl-phase0-3 / EXP-010 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7233.3 | source_pin, dataset, owned_original_log |
| artifact-016 | qwen35vl-phase0-3 / EXP-011 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7114.9 | source_pin, dataset, owned_original_log |
| artifact-017 | qwen35vl-phase0-3 / EXP-012 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7275.5 | source_pin, dataset, owned_original_log |
| artifact-018 | qwen35vl-phase0-3 / EXP-013 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7147.4 | source_pin, dataset, owned_original_log |
| artifact-019 | qwen35vl-phase0-3 / EXP-014 | qwen35_vl_35b_a3b | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8207.4 | source_pin, dataset, owned_original_log |
| artifact-020 | qwen35vl-phase0-3 / EXP-015 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 13603.2 | source_pin, dataset, owned_original_log |
| artifact-021 | qwen35vl-phase0-3 / EXP-017 | qwen35_vl_35b_a3b | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 9247.5 | source_pin, dataset, owned_original_log |
| artifact-022 | qwen35vl-phase0-3 / EXP-019 | qwen35_vl_35b_a3b | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 9761.4 | source_pin, dataset, owned_original_log |
| artifact-023 | qwen35vl-phase0-3 / EXP-027 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 13610.3 | source_pin, dataset, owned_original_log |
| artifact-024 | qwen35vl-phase0-3 / EXP-028 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 13522.8 | source_pin, dataset, owned_original_log |
| artifact-025 | qwen35vl-phase0-3 / EXP-029 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 14057.6 | source_pin, dataset, owned_original_log |
| artifact-026 | qwen35vl-phase0-3 / EXP-030 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 13736.7 | source_pin, dataset, owned_original_log |
| artifact-027 | qwen35vl-phase0-3 / EXP-031 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 13874.9 | source_pin, dataset, owned_original_log |
| artifact-028 | active-reset-20260429 / EXP-000 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5571.7 | source_pin, dataset, owned_original_log |
| artifact-029 | active-reset-20260429 / EXP-003 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5622.4 | source_pin, dataset, owned_original_log |
| artifact-030 | active-reset-20260429 / EXP-004 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5641.7 | source_pin, dataset, owned_original_log |
| artifact-031 | active-reset-20260429 / EXP-005 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5676.3 | source_pin, dataset, owned_original_log |
| artifact-032 | active-reset-20260429 / EXP-006 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5907.8 | source_pin, dataset, owned_original_log |
| artifact-033 | active-reset-20260429 / EXP-009 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14081 | source_pin, dataset, owned_original_log |
| artifact-034 | active-reset-20260429 / EXP-010 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14127.4 | source_pin, dataset, owned_original_log |
| artifact-035 | active-reset-20260429 / EXP-011 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5673.8 | source_pin, dataset, owned_original_log |
| artifact-036 | active-reset-20260429 / EXP-012 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5701.7 | source_pin, dataset, owned_original_log |
| artifact-037 | active-reset-20260429 / EXP-013 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5929.4 | source_pin, dataset, owned_original_log |
| artifact-038 | active-reset-20260429 / EXP-014 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 6459.3 | source_pin, dataset, owned_original_log |
| artifact-039 | active-reset-20260429 / EXP-015 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 5039.7 | source_pin, dataset, owned_original_log |
| artifact-040 | active-reset-20260429 / EXP-016 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 4798.6 | source_pin, dataset, owned_original_log |
| artifact-041 | active-reset-20260429 / EXP-017 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 8235.7 | source_pin, dataset, owned_original_log |
| artifact-042 | active-reset-20260429 / EXP-018 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 6543.8 | source_pin, dataset, owned_original_log |
| artifact-043 | active-reset-20260429 / EXP-019 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 12071.7 | source_pin, dataset, owned_original_log |
| artifact-044 | active-reset-20260429 / EXP-020 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 17526.6 | source_pin, dataset, owned_original_log |
| artifact-045 | active-reset-20260429 / EXP-022 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14041.9 | source_pin, dataset, owned_original_log |
| artifact-046 | active-reset-20260429 / EXP-023 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 13997.9 | source_pin, dataset, owned_original_log |
| artifact-047 | active-reset-20260429 / EXP-025 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 11922.6 | source_pin, dataset, owned_original_log |
| artifact-048 | active-reset-20260429 / EXP-026 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 26334 | source_pin, dataset, owned_original_log |
| artifact-049 | active-reset-20260429 / EXP-030 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 13641.7 | source_pin, dataset, owned_original_log |
| artifact-050 | active-reset-20260429 / EXP-034 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14306.8 | source_pin, dataset, owned_original_log |
| artifact-051 | active-reset-20260429 / EXP-035 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14285.8 | source_pin, dataset, owned_original_log |
| artifact-052 | active-reset-20260429 / EXP-036 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 12880.5 | source_pin, dataset, owned_original_log |
| artifact-053 | active-reset-20260429 / EXP-037 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14443.3 | source_pin, dataset, owned_original_log |
| artifact-054 | active-reset-20260429 / EXP-038 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 12962.2 | source_pin, dataset, owned_original_log |
| artifact-055 | active-reset-20260429 / EXP-039 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 17831.3 | source_pin, dataset, owned_original_log |
| artifact-056 | active-reset-20260429 / EXP-040fix | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 13116.5 | source_pin, dataset, owned_original_log |
| artifact-057 | active-reset-20260429 / EXP-041 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 14144.4 | source_pin, dataset, owned_original_log |
| artifact-058 | active-reset-20260429 / EXP-042 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 31439.4 | source_pin, dataset, owned_original_log |
| artifact-059 | active-reset-20260429 / EXP-043 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 32123.9 | source_pin, dataset, owned_original_log |
| artifact-060 | active-reset-20260429 / EXP-044 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 15783.4 | source_pin, dataset, owned_original_log |
| artifact-061 | active-reset-20260429 / EXP-045 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 31838.2 | source_pin, dataset, owned_original_log |
| artifact-062 | active-reset-20260429 / EXP-046 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 35844.35 | source_pin, dataset, owned_original_log |
| artifact-063 | active-reset-20260429 / EXP-047 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 42757.35 | source_pin, dataset, owned_original_log |
| artifact-064 | active-reset-20260429 / EXP-048 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 14448 | source_pin, dataset, owned_original_log |
| artifact-065 | active-reset-20260429 / EXP-050 | qwen3vl_hybrid | 64 / 512 / 65536 | iters 10-50 | accepted_measurement_artifact | unavailable / 78602.7 | source_pin, dataset, owned_original_log |
| artifact-066 | active-reset-20260429 / EXP-051 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 16329.4 | source_pin, dataset, owned_original_log |
| artifact-067 | active-reset-20260429 / EXP-052 | qwen3vl_hybrid | 64 / 512 / 65536 | iters 10-50 | accepted_measurement_artifact | unavailable / 78768.7 | source_pin, dataset, owned_original_log |
| artifact-068 | active-reset-20260429 / EXP-052v4 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 14720.7 | source_pin, dataset, owned_original_log |
| artifact-069 | active-reset-20260429 / EXP-052v5 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 15284.7 | source_pin, dataset, owned_original_log |
| artifact-070 | active-reset-20260429 / EXP-052v6 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 19015.7 | source_pin, dataset, owned_original_log |
| artifact-071 | active-reset-20260429 / EXP-052v7 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 23585 | source_pin, dataset, owned_original_log |
| artifact-072 | active-reset-20260429 / EXP-053 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 17829.7 | source_pin, dataset, owned_original_log |
| artifact-073 | active-reset-20260429 / EXP-054 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 27097.2 | source_pin, dataset, owned_original_log |
| artifact-074 | active-reset-20260429 / EXP-054final2 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14320.5 | source_pin, dataset, owned_original_log |
| artifact-075 | active-reset-20260429 / EXP-055fix | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14484.3 | source_pin, dataset, owned_original_log |
| artifact-076 | active-reset-20260429 / EXP-056fix | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14559.8 | source_pin, dataset, owned_original_log |
| artifact-077 | active-reset-20260429 / EXP-057 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 14607.4 | source_pin, dataset, owned_original_log |
| artifact-078 | active-reset-20260429 / EXP-057fix | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 17197.1 | source_pin, dataset, owned_original_log |
| artifact-079 | active-reset-20260429 / EXP-058 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 17447.6 | source_pin, dataset, owned_original_log |
| artifact-080 | active-reset-20260429 / EXP-059 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 22661.6 | source_pin, dataset, owned_original_log |
| artifact-081 | active-reset-20260429 / EXP-059fix | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14781.2 | source_pin, dataset, owned_original_log |
| artifact-082 | active-reset-20260429 / EXP-060 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 21716.5 | source_pin, dataset, owned_original_log |
| artifact-083 | active-reset-20260429 / EXP-060fix | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 15080.2 | source_pin, dataset, owned_original_log |
| artifact-084 | active-reset-20260429 / EXP-061 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 25262.5 | source_pin, dataset, owned_original_log |
| artifact-085 | active-reset-20260429 / EXP-062 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 18005.2 | source_pin, dataset, owned_original_log |
| artifact-086 | active-reset-20260429 / EXP-064 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 25716.6 | source_pin, dataset, owned_original_log |
| artifact-087 | active-reset-20260429 / EXP-065 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 30494.7 | source_pin, dataset, owned_original_log |
| artifact-088 | active-reset-20260429 / EXP-070 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 7870.3 | source_pin, dataset, owned_original_log |
| artifact-089 | active-reset-20260429 / EXP-071v3 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 12849 | source_pin, dataset, owned_original_log |
| artifact-090 | active-reset-20260429 / EXP-072 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8492.8 | source_pin, dataset, owned_original_log |
| artifact-091 | active-reset-20260429 / EXP-073v4 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 13120.9 | source_pin, dataset, owned_original_log |
| artifact-092 | active-reset-20260429 / EXP-074 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 14605.8 | source_pin, dataset, owned_original_log |
| artifact-093 | active-reset-20260429 / EXP-075v2 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 12940.2 | source_pin, dataset, owned_original_log |
| artifact-094 | active-reset-20260429 / EXP-076v2 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 7898.4 | source_pin, dataset, owned_original_log |
| artifact-095 | active-reset-20260429 / EXP-077v8 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 18222.9 | source_pin, dataset, owned_original_log |
| artifact-096 | active-reset-20260429 / EXP-078v7 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 22086.1 | source_pin, dataset, owned_original_log |
| artifact-097 | active-reset-20260429 / EXP-079v4 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 23821.8 | source_pin, dataset, owned_original_log |
| artifact-098 | active-reset-20260429 / EXP-080v2 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 21897.1 | source_pin, dataset, owned_original_log |
| artifact-099 | active-reset-20260429 / EXP-080v3 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 25362 | source_pin, dataset, owned_original_log |
| artifact-100 | active-reset-20260429 / EXP-083 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 30574.6 | source_pin, dataset, owned_original_log |
| artifact-101 | active-reset-20260429 / EXP-084 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 28523.6 | source_pin, dataset, owned_original_log |
| artifact-102 | active-reset-20260429 / EXP-085 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 26482 | source_pin, dataset, owned_original_log |
| artifact-103 | active-reset-20260429 / EXP-090 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 25778.7 | source_pin, dataset |
| artifact-104 | active-reset-20260429 / EXP-092 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 28645.7 | source_pin, dataset |
| artifact-105 | active-reset-20260429 / EXP-093 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 29205.2 | source_pin, dataset |
| artifact-106 | active-reset-20260429 / EXP-094 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 29468.1 | source_pin, dataset |
| artifact-107 | active-reset-20260429 / EXP-095-mock | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 30386.5 | source_pin, dataset |
| artifact-108 | active-reset-20260429 / EXP-096-mock | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 25822.6 | source_pin, dataset |
| artifact-109 | active-reset-20260429 / EXP-099 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 51933.2 | source_pin, dataset |
| artifact-110 | active-reset-20260429 / EXP-107 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 36753.9 | source_pin, dataset |
| artifact-111 | active-reset-20260429 / EXP-108-partial | qwen3vl_hybrid | None / 512 / 8192 | null | recorded_partial_verifier_skipped | unavailable / None | world, config, owned_original_log |
| artifact-112 | active-reset-20260429 / EXP-109-partial | qwen3vl_hybrid | None / 512 / 8192 | null | recorded_partial_verifier_skipped | unavailable / None | world, config, owned_original_log |
| artifact-113 | active-reset-20260429 / EXP-110 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 43765.7 | source_pin |
| artifact-114 | active-reset-20260429 / EXP-115 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 29843.7 | source_pin, dataset |
| artifact-115 | active-reset-20260429 / EXP-116 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 42351 | source_pin, dataset |
| artifact-116 | active-reset-20260429 / EXP-117 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 41108.8 | source_pin, dataset |
| artifact-117 | active-reset-20260429 / EXP-118 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 34720.8 | source_pin |
| artifact-118 | active-reset-20260429 / EXP-119 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 34952.8 | source_pin |
| artifact-119 | active-reset-20260429 / EXP-121 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 30781.8 | source_pin, dataset |
| artifact-120 | active-reset-20260429 / EXP-123 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 30178.5 | source_pin, dataset |
| artifact-121 | active-reset-20260429 / EXP-124 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 30510.5 | source_pin, dataset |
| artifact-122 | active-reset-20260429 / EXP-125 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 35031.7 | source_pin |
| artifact-123 | active-reset-20260429 / EXP-126 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 34833 | source_pin |
| artifact-124 | active-reset-20260429 / EXP-133 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 39248.4 | source_pin, owned_original_log |
| artifact-125 | active-reset-20260429 / EXP-134 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 39167.4 | source_pin, owned_original_log |
| artifact-126 | active-reset-20260429 / EXP-135 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 34612.5 | source_pin, owned_original_log |
| artifact-127 | active-reset-20260429 / EXP-136 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8623.8 | source_pin, owned_original_log |
| artifact-128 | active-reset-20260429 / EXP-137 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8573.6 | source_pin, owned_original_log |
| artifact-129 | active-reset-20260429 / EXP-138 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 16604.5 | source_pin, owned_original_log |
| artifact-130 | active-reset-20260429 / EXP-140 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8642.3 | source_pin, owned_original_log |
| artifact-131 | active-reset-20260429 / EXP-141 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8623.6 | source_pin, owned_original_log |
| artifact-132 | active-reset-20260429 / EXP-142 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8600.1 | source_pin, owned_original_log |
| artifact-133 | active-reset-20260429 / EXP-143 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 9798.4 | source_pin, owned_original_log |
| artifact-134 | active-reset-20260429 / EXP-152 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 40201.1 | source_pin, owned_original_log |
| artifact-135 | active-reset-20260429 / EXP-153 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 39094.8 | source_pin, owned_original_log |
| artifact-136 | active-reset-20260429 / EXP-154 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 39059.1 | source_pin, owned_original_log |
| artifact-137 | active-reset-20260429 / EXP-160 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 43294.6 | source_pin, owned_original_log |
| artifact-138 | active-reset-20260429 / EXP-161 | qwen3vl_hybrid | 128 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 70202.5 | source_pin, owned_original_log |
| artifact-139 | active-reset-20260429 / EXP-162 | qwen3vl_hybrid | 128 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 69757.1 | source_pin, owned_original_log |
| artifact-140 | active-reset-20260429 / EXP-163 | qwen3vl_hybrid | 128 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 79219.1 | source_pin, owned_original_log |
| artifact-141 | active-reset-20260429 / EXP-164 | qwen3vl_hybrid | 256 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 117029.9 | source_pin, owned_original_log |
| artifact-142 | active-reset-20260429 / EXP-165 | qwen3vl_hybrid | 256 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 116344.2 | source_pin, owned_original_log |
| artifact-143 | active-reset-20260429 / EXP-172 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 16590.9 | source_pin, owned_original_log |
| artifact-144 | active-reset-20260429 / EXP-180 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 16261.4 | source_pin, owned_original_log |
| artifact-145 | active-reset-20260429 / EXP-182 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 16264.8 | source_pin, owned_original_log |
| artifact-146 | active-reset-20260429 / EXP-185 | qwen3vl_hybrid | 64 / 512 / 65536 | iters 10-50 | accepted_measurement_artifact | unavailable / 163754.3 | source_pin, owned_original_log |
| artifact-147 | active-reset-20260429 / EXP-200 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8590.2 | source_pin, owned_original_log |
| artifact-148 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 9005.8 | source_pin, owned_original_log |
| artifact-149 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8997.7 | source_pin, owned_original_log |
| artifact-150 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8992.3 | source_pin, owned_original_log |
| artifact-151 | active-reset-20260429 / EXP-207 | null | None / None / None | null | recorded_failed | unavailable / None | model, world, gbs, config, owned_original_log, sequence_length |
| artifact-152 | active-reset-20260429 / EXP-208 | null | None / None / None | null | recorded_failed | unavailable / None | model, world, gbs, config, owned_original_log, sequence_length |
| artifact-153 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 9457.1 | source_pin, owned_original_log |
| artifact-154 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 9436.5 | source_pin, owned_original_log |
| artifact-155 | active-reset-20260429 / EXP-212 | null | None / None / None | null | recorded_failed | unavailable / None | model, world, gbs, config, owned_original_log, sequence_length |
| artifact-156 | active-reset-20260429 / EXP-213 | null | None / None / None | null | recorded_failed | unavailable / None | model, world, gbs, config, owned_original_log, sequence_length |
| artifact-157 | active-reset-20260429 / EXP-230 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 8997.7 | source_pin, owned_original_log |
| artifact-158 | active-reset-20260429 / EXP-231 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 21999.4 | source_pin, owned_original_log |
| artifact-159 | active-reset-20260429 / EXP-232 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 39437.9 | source_pin, owned_original_log |
| artifact-160 | active-reset-20260429 / EXP-233 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 21879.8 | source_pin, owned_original_log |
| artifact-161 | active-reset-20260429 / EXP-234 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 15493.8 | source_pin, owned_original_log |
| artifact-162 | active-reset-20260429 / EXP-235 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 16037.8 | source_pin, owned_original_log |
| artifact-163 | active-reset-20260429 / EXP-236 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 39472.2 | source_pin, owned_original_log |
| artifact-164 | active-reset-20260429 / EXP-237 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 21755.2 | source_pin, owned_original_log |
| artifact-165 | active-reset-20260429 / EXP-238 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 16106.6 | source_pin, owned_original_log |
| artifact-166 | active-reset-20260429 / EXP-239 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 17300.3 | source_pin, owned_original_log |
| artifact-167 | active-reset-20260429 / EXP-240 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 26962.8 | source_pin, owned_original_log |
| artifact-168 | active-reset-20260429 / EXP-241 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 16635.4 | source_pin, owned_original_log |
| artifact-169 | active-reset-20260429 / EXP-246 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 21298.8 | source_pin, owned_original_log |
| artifact-170 | active-reset-20260429 / EXP-247 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 36418.5 | source_pin, owned_original_log |
| artifact-171 | active-reset-20260429 / EXP-248 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 20604 | source_pin, owned_original_log |
| artifact-172 | active-reset-20260429 / EXP-249 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 35310.2 | source_pin, owned_original_log |
| artifact-173 | active-reset-20260429 / EXP-250 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 10447.2 | source_pin, owned_original_log |
| artifact-174 | active-reset-20260429 / EXP-251 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 10350.8 | source_pin, owned_original_log |
| artifact-175 | active-reset-20260429 / EXP-254 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 40251.2 | source_pin, owned_original_log |
| artifact-176 | active-reset-20260429 / EXP-255 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 26562.5 | source_pin, owned_original_log |
| artifact-177 | active-reset-20260429 / EXP-256 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 44493.4 | source_pin, owned_original_log |
| artifact-178 | active-reset-20260429 / EXP-257 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 29317.1 | source_pin, owned_original_log |
| artifact-179 | active-reset-20260429 / EXP-258 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 45307.5 | source_pin, owned_original_log |
| artifact-180 | active-reset-20260429 / EXP-259 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 28021.9 | source_pin, owned_original_log |
| artifact-181 | active-reset-20260429 / EXP-261 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 12263.3 | source_pin, owned_original_log |
| artifact-182 | active-reset-20260429 / EXP-263 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 33465.9 | source_pin, owned_original_log |
| artifact-183 | active-reset-20260429 / EXP-264 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 36513 | source_pin, owned_original_log |
| artifact-184 | active-reset-20260429 / EXP-265 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 28602.9 | source_pin, owned_original_log |
| artifact-185 | active-reset-20260429 / EXP-PR131-ENCODER-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 4170.2 | source_pin, dataset |
| artifact-186 | active-reset-20260429 / EXP-PR131-ENCODER-CP1-PAIR749724-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 4358.8 | source_pin, dataset |
| artifact-187 | active-reset-20260429 / EXP-PR131-ENCODER-CP2-PAIR749724-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 4538.4 | source_pin, dataset |
| artifact-188 | active-reset-20260429 / EXP-PR131-IMAGE1X-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 4144.3 | source_pin, dataset |
| artifact-189 | active-reset-20260429 / EXP-PR131-IMAGE1X-CP2-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 4293.5 | source_pin, dataset |
| artifact-190 | active-reset-20260429 / EXP-PR131-IMAGE2X-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 4151.5 | source_pin, dataset |
| artifact-191 | active-reset-20260429 / EXP-PR131-IMAGE2X-CP2-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 4267.2 | source_pin, dataset |
| artifact-192 | active-reset-20260429 / EXP-PR131-IMAGE4X-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 4574.4 | source_pin, dataset |
| artifact-193 | active-reset-20260429 / EXP-PR131-IMAGE4X-CP2-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 4806.8 | source_pin, dataset |
| artifact-194 | active-reset-20260429 / EXP-PR131-IMAGE8X-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 5571.4 | source_pin, dataset |
| artifact-195 | active-reset-20260429 / EXP-PR131-IMAGE8X-CP2-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 6054.7 | source_pin, dataset |
| artifact-196 | active-reset-20260429 / EXP-PR131-MANTIS16-752807-20-GB300-legacy-window | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 7221.1 | source_pin |
| artifact-197 | active-reset-20260429 / EXP-PR131-MANTIS16-752807-20-GB300 | qwen3vl | 16 / 64 / 16384 | [4, 20] | accepted_measurement_artifact | unavailable / 7221.1 | source_pin |
| artifact-198 | active-reset-20260429 / EXP-PR131-TFLOPS16-SEQ4096-GBS256 | qwen3vl | 16 / 256 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 7296 | source_pin |
| artifact-199 | active-reset-20260429 / EXP-PR7-FOURWAY-751915-pr7_baseline | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 14961 | source_pin, dataset |
| artifact-200 | active-reset-20260429 / EXP-PR7-FOURWAY-752159-pr131_latest | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 8901.4 | source_pin, dataset |
| artifact-201 | active-reset-20260429 / EXP-PR7-FOURWAY-752159-pr7_baseline | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 13804.5 | source_pin, dataset |
| artifact-202 | active-reset-20260429 / EXP-PR7-FOURWAY-752159-pr7_mdp | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 11479.4 | source_pin, dataset |
| artifact-203 | active-reset-20260429 / EXP-PR7-FOURWAY-752159-pr7_packing | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 9820.5 | source_pin, dataset |
| artifact-204 | active-reset-20260429 / EXP-bridge1 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14362.6 | source_pin, dataset, owned_original_log |
| artifact-205 | active-reset-20260429 / EXP-bridge2 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14587.7 | source_pin, dataset, owned_original_log |
| artifact-206 | active-reset-20260429 / EXP-va1 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 21436.9 | source_pin, dataset, owned_original_log |
| artifact-207 | active-reset-20260429 / EXP-va2 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 36553.8 | source_pin, dataset, owned_original_log |
| artifact-208 | active-reset-20260429 / EXP-va3 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 24283.3 | source_pin, dataset, owned_original_log |
| artifact-209 | active-reset-20260429 / EXP-va4 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14570.9 | source_pin, dataset, owned_original_log |
| artifact-210 | active-reset-20260429 / EXP-va5 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 25380.7 | source_pin, dataset, owned_original_log |
| artifact-211 | active-reset-20260429 / EXP-va6 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 39772.5 | source_pin, dataset, owned_original_log |
| artifact-212 | active-reset-20260429 / EXP-varimg2 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | unavailable / 14669.1 | source_pin, dataset, owned_original_log |
| artifact-213 | active-reset-20260429 / EXP-varlen3 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 19456.8 | source_pin, dataset, owned_original_log |
| artifact-214 | active-reset-20260429 / EXP-varlen6 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 21527.7 | source_pin, dataset, owned_original_log |
| artifact-215 | active-reset-20260429 / EXP-vb1 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 22704.8 | source_pin, dataset, owned_original_log |
| artifact-216 | active-reset-20260429 / EXP-vb2 | qwen3vl_hybrid | 64 / 512 / 24576 | iters 10-50 | accepted_measurement_artifact | unavailable / 27402.5 | source_pin, dataset, owned_original_log |
| artifact-217 | active-reset-20260429 / EXP-vb3 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 33112.4 | source_pin, dataset, owned_original_log |
| artifact-218 | active-reset-20260429 / EXP-vb4 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 36434.1 | source_pin, dataset, owned_original_log |
| artifact-219 | active-reset-20260429 / EXP-vb5 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 46711 | source_pin, dataset, owned_original_log |
| artifact-220 | active-reset-20260429 / EXP-vb6 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 27917.1 | source_pin, dataset, owned_original_log |
| artifact-221 | active-reset-20260429 / EXP-vc1 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 22355.7 | source_pin, dataset, owned_original_log |
| artifact-222 | active-reset-20260429 / EXP-vc3 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 43032 | source_pin, dataset, owned_original_log |
| artifact-223 | active-reset-20260429 / EXP-vc4 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | unavailable / 18243.3 | source_pin, dataset, owned_original_log |
| artifact-224 | active-reset-20260429 / EXP-vc5 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 22295.7 | source_pin, dataset, owned_original_log |
| artifact-225 | active-reset-20260429 / EXP-vc6 | qwen3vl_hybrid | 64 / 512 / 24576 | iters 10-50 | accepted_measurement_artifact | unavailable / 41377.8 | source_pin, dataset, owned_original_log |
| artifact-226 | active-reset-20260429 / EXP-vd1 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 26621.8 | source_pin, dataset, owned_original_log |
| artifact-227 | active-reset-20260429 / EXP-vd2 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 27288.4 | source_pin, dataset, owned_original_log |
| artifact-228 | active-reset-20260429 / EXP-vd3 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 38304 | source_pin, dataset, owned_original_log |
| artifact-229 | active-reset-20260429 / EXP-vd4 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | unavailable / 51145.1 | source_pin, dataset, owned_original_log |
| artifact-230 | active-reset-20260429 / EXP-vd5 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | unavailable / 28451.8 | source_pin, dataset, owned_original_log |
| artifact-231 | active-reset-20260429 / EXP-vd6 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | unavailable / 25960.6 | source_pin, dataset, owned_original_log |

</details>
