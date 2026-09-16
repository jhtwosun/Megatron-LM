# Entire owned-campaign catalog — publication draft

Scope explicitly includes the entire owned campaign, including archived Qwen3.5-VL phases0–3 and archived Qwen3 phase4, the April29 numbering reset, subsequent text/hybrid/real-data work, and the current PR131/GB300 experiments. Current-thread publication documents remain intact. No peer worktree, scheduler query, corpus fetch, runtime change, or external publication was used.

This is a metadata and documentary evidence catalog, not a completed accounting correction or a claim that every historical execution can be reconstructed. The campaign-local machine inventory and generator are not bundled here because they retain private evidence paths and documentary excerpts. This public summary preserves namespaces, counts and unavailable proof without exposing those records.

## Coverage and counting

The counts below describe the retained 231-artifact historical inventory. New accepted fixed-input sweep results are published separately; 231 is not the total count of all current results. Historical corrected rates remain unavailable rather than being inferred from newer inputs or reused geometry.

See the [new corrected-accounting snapshot](sweep-corrected-snapshot-20260916.md) for separately accepted formal cells and current incomplete configurations.

| Namespace | Result-location JSONs | Experiment directories | Directories unlinked to result config references |
|---|---:|---:|---:|
|Active campaign after April29 numbering reset|204|315|132|
|Archived qwen35vl-phase0-3|22|28|6|
|Archived qwen3-phase4-final|5|8|3|
|Total|231|351|141|

The231 JSONs are **not231 successful experiments**. They contain225 measurement artifacts,4explicit FAILED records (EXP207/208/212/213), and2partial/verifier-skipped records (EXP108-partial/109-partial). Mantis752807 primary and supplemental JSONs describe one attempt. After that known alternate representation,224 provisional measurement attempt/cell groups remain; adding the six explicit failure/partial records gives230 result-linked attempt groups. Grouping uses namespace plus recorded log reference, not experiment number alone. Empty/missing log references remain unresolved identities. These are not a census of all scheduler jobs: one job can contain multiple cells, retries may be documented only in prose, and copied/renamed logs can conceal duplication.

The141 unlinked directories are **unknown status**, not inferred failures. Some contain qualifications, variants, proposals, or successful runs not linked by the archived config path.345 of351 directories contain immediate YAML/shell configuration files. The active baseline directory contains no substantive files (only its placeholder). This does not mean there were no baseline experiments: baseline roles occur in experiment records and reports.

The scan indexes540 failure/cancellation or qualification/diagnostic documentary mentions across STATUS,5791-line journal, archive snapshots, reports and immediate experiment Markdown. Each entry has a source and line reference, extracted experiment IDs and provisional six-digit job candidates. These are repeated mentions, not540 failed jobs. A sentence can describe both a past failure and later success; six-digit numbers can be unrelated quantities. No categorical terminal status is inferred from the mention index alone.

## Model, hardware and dataset coverage

| Recorded model label | JSON count | Confidence |
|---|---:|---|
|qwen35_vl_35b_a3b|22|Recorded archived model label|
|qwen3_30b_a3b|5|Recorded archived text-model label|
|qwen3vl_hybrid|175|Recorded label, including two partial records|
|qwen3|6|Recorded active labels: EXP070,071v3,072,073v4,075v2,076v2|
|qwen3vl|19|Current-thread records, including alternate Mantis window representation|
|Unspecified|4|Explicit failure JSONs lack a model field; do not manufacture one|

Names are not canonical architecture equivalence. Original configs, model factories, executed commits, resolved vocabulary and attention/MoE variants must be checked before combining labels or formulas. Archived qwen3-phase4 and active qwen3 are separate namespaces even when an EXP number repeats.

Recorded config/recipe products suggest201 rows at world64,19 at world16,3 at128,2 at256, and6 without a complete product. **The128/256 values are not confirmed hardware counts:** EXP161–163 encode TP1/PP1/CP4/DP32, and EXP164–165 CP8/DP32. This may reflect stale DP, dynamic-CP semantics, or distinct topology. Keep these explicitly suspect until original launch/accounting evidence resolves them. Do not claim the campaign actually used128/256 GPUs from this product alone.

GBS extraction finds208 rows at512,18 at64,1 at256, and4 unavailable. These are artifact counts including alternate/failure records, not distinct experiments. Hardware inference from GB300 current-thread names versus the historical GB200 campaign default is recorded with confidence flags; historical GB200 is **provisional campaign context**, not a per-attempt hardware receipt. STATUS's current GB300 override must not relabel old GB200 runs.

Dataset metadata is incomplete:160 records have no provider in the parsed result/config;61 say `real`,2partial records explicitly name real Mantis multi_vqa,4failed records specify multi_vqa or four-subset Mantis,2Mantis representations say Energon, and1each explicitly say mock/mock_mdp. Missing provider is not permission to assume mock. Older real-data experiments include variable-length/image-count and multi-subset paths whose consumed distribution cannot be recovered from a provider label. Current Mantis/Nemotron preparation evidence is separate from historical Mantis slices.

## Evidence availability and correction readiness

Only38 result references currently resolve to a raw log **inside the owned campaign tree**, counting both Mantis representations. Other logs are often historical absolute SLURM paths outside this tree; they were neither searched nor read in this bounded task. Mark them unavailable in this inventory, not deleted. All231 JSONs were readable. Source/geometry/audit candidates appear immediately in seven current-thread experiment roots; older experiment directories mostly retain configs/launchers, not an immediately named source or consumed-boundary manifest. Nested and external artifacts may exist: absence from this immediate-file candidate index is not proof of absence everywhere.

| Family | Confirmed retained evidence | Unavailable or still provisional for correction |
|---|---|---|
|Archived Qwen3.5-VL phases0–3|22 result JSONs,28 experiment directories/configs, archive README/reports, historical tag context|Executed source and exact native formula, full resolved model/tokenizer, raw timing-window verification, valid attended boundaries and consumed vision geometry not established by this scan|
|Archived Qwen3 phase4|5 result JSONs,8 directories/configs, STATUS/journal snapshots and archive README|Same executed-source/formula/window proof gap; text-only simplifies vision accounting but does not prove CP/padding/MoE FLOPs|
|Active reset hybrid/text and historical real-data work|Result metadata and extensive configs, STATUS/journal/report references; some source heads in partial/failure records|Per-attempt source/model identity, dynamic CP/DP semantics, actual hardware, original raw logs, tokenizer/data revisions and final boundaries require individual audit|
|Current reference CP/sweep/TF|Sealed measured sources, resolved model/function hashes, original timings and content geometry, reviewed gates|Final native attention-boundary replay still pending; content moments alone do not establish attended padded work|
|Current native four-way|Sealed PR7/PR131 sources, original timings, executed configuration and deterministic generator|No original per-step content geometry or consumed hashes; exact historical replay identity remains unproved|
|Current Mantis752807|Pinned prepared corpus/tokenizer, original timings, all20 content geometry records, complete model gates|No archived final logical/physical attention boundaries or exact consumed-record manifest; runtime replay required|
|Current Nemotron preparation|All-row CPU token and bounded pack/pixel proof|No full-model training or performance result to correct|

PR131's extracted native FLOP function equality proof applies only to its audited current source families, not automatically to archived Qwen3.5-VL, earlier hybrid source, FP8, MTP, FSDP, or dynamic-attention variants. Archived scalar TFLOPs cannot be multiplied by an arbitrary universal factor. Every future corrected row must retain its original per-step timing window, executed function/source hash, all accessed resolved arguments, exact input/boundary semantics, and confidence in historical consumption identity. Unknown quantities remain null. No numerical campaign-wide corrections were executed.

## Recorded unsuccessful and qualification attempts

Machine records preserve all four explicit failed JSONs and both partial JSONs separately from measurements. The documentary index additionally covers historical OOM, collective failures, configuration probes, salloc qualification and retries, with line-level evidence and unresolved identity. Recent attempts are already independently resolved:751915 MDP watchdog after8;752159 outer timeout after four complete measurements and two complete profiles; missing packing/latest profile results;752462 training10 then validation failure;752786 clean eval0 qualification;752807 completed formal20. Those receipts do not retroactively validate older prose-only attempts.

The current catalog is deliberately conservative. A complete unique-attempt ledger requires reconciling documentary job candidates against original submit/terminal receipts, including archived retries and multi-cell jobs. The correct immediate output is the counts and gaps above, not an invented exact total of all experiments ever attempted.

## Verification and next review boundary

Read-only metadata parse completed with0 JSON read errors. Numbering-reset namespaces remain explicit; Mantis alternate-window duplication is identified; failed/partial artifacts are excluded from accepted measurement counts. This summary does not claim a fully reconciled per-attempt public ledger. Reviewed archive/source/log recovery can fill the gaps incrementally; preserve original rates and unavailable corrected values until then.
