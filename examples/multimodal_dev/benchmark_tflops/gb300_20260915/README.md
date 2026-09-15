# GB300 experiment and accounting supplement — 2026-09-15

**Working draft. Corrected accounting is not yet established.** This catalog preserves historical observed timings and legacy native TFLOPs estimates. Every corrected-rate entry is `null` until the executed formula, resolved arguments, consumed inputs, and final attention boundaries have adequate proof. Null means unavailable, not zero or no work.

This is a new current-thread supplement on PR131 head `bbba1001543a9c99a26ae8a8370b40e005c80d80`. It does not replace the separate historical ledger in [NATIVE_ACCOUNTING.md](../NATIVE_ACCOUNTING.md), and does not import that ledger's experiment inputs. Measured source bases and adaptations are distinct from this documentation branch.

## Reading guide

- [Entire owned-campaign catalog](campaign-wide-catalog.md): archived and active namespaces,231 result artifacts,351 experiment directories, duplicate/failure/unknown distinctions and correction gaps.
- [Per-experiment recorded details](per-experiment.md) and [machine-readable fields](per-experiment.json): all 231 artifacts, recorded stack/recipe fields, provisional attempt groups, fingerprints, and missing references. These are descriptive records, not 231 newly verified successes; five ambiguous world-size products are explicitly flagged.
- [Data distributions and configurations](data-and-configs.md): generator versus prepared-corpus versus consumed-geometry evidence, source/data pins, and known gaps.
- [Evidence fingerprints](evidence-fingerprints.md): retained result and raw-log SHA-256 values. Raw logs, media, and private filesystem paths are intentionally not bundled.
- [Completed four-way profile evidence](profile-summary.md): finalized packing/latest continuation, process-aware phase observations, and explicit capture/configuration limitations. Diagnostic only, not a speed ranking.
- [Scoped accounting utilities](accounting-utilities.md): seven standard-library tests and native formula equivalence checks; historical native replay remains unexecuted. Tool availability does not establish corrected rates or historical boundary identity.
- [Ongoing qualification status](24h-progress.md): eight-cell CPU gate, low-work model qualification, high-count OOM, and unstarted cells. No new sweep performance result is claimed.

## Accepted historical records

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
|Fixed sweep and TF750157|Logged content moments, fixed-grid source, original per-step timings|Native replay of all consumed DP streams; final logical and physical cumulative sequence boundaries; valid-attention semantics; source/argument identity|
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

This package is a transparent draft, not a completed accounting correction. No measured source, original accepted JSON, historical ledger, or raw log is overwritten. Corrected columns remain null pending proof and independent review. No checkpoint/model-weight download, new training run, or external publication is implied by these documents.
