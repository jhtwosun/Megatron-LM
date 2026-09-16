# GB300 experiment results and accounting — 2026-09-16

**Read first.** This Draft is the results/reporting supplement to PR131, based on `bbba1001543a9c99a26ae8a8370b40e005c80d80`. It covers this team's own campaign: 231 historical result artifacts, six newly accepted fixed-input cells, five older encoder-CP pairs, four-way measurements/profiles, real Mantis and two loader run orders. These are overlapping evidence groups, **not additive counts of successful experiments**. Historical failures, incomplete transfers and unavailable corrections remain explicit.

The report uses [PR7](https://github.com/jhtwosun/Megatron-LM/pull/7) as a presentation reference; results below retain their own sources and evidence. This documentation branch does not imply that every measured source adaptation is contained in PR131 or this PR. The separate [native accounting ledger](../NATIVE_ACCOUNTING.md) retains its own scope.

## New 20-step reruns: PR131-native components and all-rank memory

These reruns retain the original four mock arms and separately preserved Mantis256 (236 train / 20 validation examples; eval0), not the later loader-serialization candidate. New isolated overlays reuse **exact PR131 `bbba100` `workload_flops.py` and `collect_workload.py`**, plus all-rank allocator recording. Measured model sources remain PR7 `e1484af4` and older PR131 `57a5c223`; the reporting backport is not a model/source-family replacement. Each formal cell uses 20 total steps, LR warmup/decay 2/20, world16/GBS64/MBS1 and primary iterations 4–20; supplemental 10–20 is retained separately. Do not compare these windows/horizons to the older 50-step table as an isolated speedup.

**Accounting follows PR131 itself.** Encoder, useful-content decoder and their sum are arithmetic means of the native per-step component rates over the whole VLM optimizer-step duration. The separate raw Megatron fixed-shape decoder rate is retained, not substituted for useful-content decoder. Prepadding content T/U and global image-grid R/A use PR131's count-normalized aggregation; no custom final-CU correction is applied. The native helper excludes padding/non-matmul work and uses the 3×forward convention, so these are modeled rates, not executed hardware FLOPs or utilization. Peak memory is the maximum across **all 16 ranks**, allocator lifetime including initialization and training, without resets; allocated and reserved are distinct decimal GB.

| Job | Original workload / mode | Status | Median step ms | Scheduled tok/s global (per GPU) | PR131 workload encoder TF/GPU | PR131 workload decoder TF/GPU | PR131 workload total TF/GPU | Megatron native fixed-shape decoder TF/GPU | Peak allocated / reserved GB |
|---|---|---|---|---|---|---|---|---|---|
| 759791 | PR7 ordinary / MDP off | Completed 20; verified cell | 13374.4 | 78401.7 (4900.1) | 23.092 | 88.958 | 112.051 | 178.435 | 162.536 / 187.102 |
| 759791 | PR7 MDP nonfused | Completed 20; verified cell | 10643.4 | 98518.9 (6157.4) | 28.949 | 111.495 | 140.444 | 223.647 | 201.152 / 203.705 |
| 759791 | PR7 fused-window retain | Completed 20; verified cell | 8116.1 | 129197.0 (8074.8) | 38.552 | 148.500 | 187.052 | 297.876 | 202.550 / 241.433 |
| 759791 | PR131 fused-window retain | Completed 20; verified cell | 8128.1 | 129006.3 (8062.9) | 40.167 | 154.736 | 194.902 | 310.376 | 233.772 / 235.101 |
| 759796 | Original Mantis256 / PR131 fused | Completed 20; verified cell | 8677.1 | 120844.1 (7552.8) | 22.967 | 99.693 | 122.660 | 281.282 | 181.995 / 183.744 |

Both formal jobs completed successfully. Consumed geometry is not identical across all modes: rates and timings are descriptive, not an isolated MDP/packing gain; interpret each row with its actual consumed work. Mantis is a different dataset and is not a fifth matched mock arm. Modeled rates are not hardware utilization. The PR7 fused row has higher observed allocator reservation than its nonfused row; reserved memory is not live tensor allocation. No repeated randomized causal estimate is claimed from this ordered rerun.

## Fixed-input MDP ON/OFF: six completed matched-shape pairs

Original ON job **758245** and new OFF job **759971** both completed all six 20-step cells. Primary window is **4–20** (17 samples), world16/GBS64/MBS1/sequence16384; counts are images per raw document, with four documents per packed bin. Every paired value below is ordered **ON / OFF**. Component TF/GPU means TFLOPs/GPU/s from the exact PR131 native helper; step times are medians. Scheduled token rates are capacity-normalized, not useful-content tokens.

| Images × side | Step ms ON / OFF | OFF step change | Scheduled tok/s global (per GPU), ON / OFF | Encoder TF/GPU ON / OFF | Decoder TF/GPU ON / OFF | Total TF/GPU ON / OFF | Megatron fixed decoder TF/GPU ON / OFF | OFF peak allocated / reserved GB |
|---|---|---|---|---|---|---|---|---|
| 1 × 224 | 6475.1 / 6781.7 | +4.74% | 161939.7 (10121.2) / 154618.5 (9663.7) | 1.253 / 1.188 | 243.955 / 231.303 | 245.207 / 232.491 | 390.135 / 369.900 | 79.820 / 81.856 |
| 1 × 448 | 6586.1 / 6997.7 | +6.25% | 159210.5 (9950.7) / 149845.8 (9365.4) | 5.365 / 5.023 | 240.813 / 225.473 | 246.178 / 230.496 | 385.106 / 360.582 | 85.970 / 87.510 |
| 1 × 896 | 7245.8 / 7426.2 | +2.49% | 144715.0 (9044.7) / 141199.5 (8825.0) | 25.397 / 24.786 | 217.270 / 212.041 | 242.667 / 236.826 | 347.453 / 339.088 | 110.569 / 112.091 |
| 2 × 448 | 6702.8 / 7108.1 | +6.05% | 156438.5 (9777.4) / 147518.5 (9219.9) | 10.513 / 9.928 | 235.947 / 222.823 | 246.459 / 232.750 | 377.318 / 356.329 | 94.169 / 95.657 |
| 4 × 448 | 6955.6 / 7343.4 | +5.58% | 150752.8 (9422.0) / 142791.6 (8924.5) | 20.132 / 19.180 | 225.920 / 215.239 | 246.051 / 234.419 | 361.288 / 344.206 | 110.568 / 111.860 |
| 8 × 448 | 7966.6 / 8733.2 | +9.62% | 131621.5 (8226.3) / 120067.8 (7504.2) | 35.337 / 32.127 | 198.278 / 180.269 | 233.615 / 212.396 | 317.082 / 288.282 | 143.369 / 144.236 |

**Input and stack evidence.** All six formal pairs passed exact ordered 320-bin-per-logical-DP descriptor and input/label/loss-mask/position tensor-hash comparisons, validating all16 ON producer replicas and8 OFF PP0 producers. PP1 delivery is source-proved broadcast, not independently recorded consumer hashes; unrecorded pixel bytes are not asserted. All17 primary T/U/R/A rows match in each pair. All770 logged runtime fields were compared per pair: only MDP enabled, fused window, its sequence cap (131072→0), explicit provider routing (`mock`→`mock_mdp`) and the output directory differ; the other765 fields match. OFF additionally includes the new allocator-memory hook. ON all-rank lifetime memory is unavailable, not zero.

**Analysis.** OFF step medians are higher in all six observations (+2.49% to +9.62%); this supports further testing of the combined MDP/fused path, not an isolated MDP or fusion effect. The difference is not monotonic in image side/count. Separate allocations, instrumentation differences and one ordered observation per cell preclude a stable causal winner or variance claim. Rates use whole-step time and native modeled work, not hardware utilization.

**Failure disposition.** The initial OFF qualification selected the wrong generic-mock route and failed before iteration1. Explicit `mock_mdp` preserved the original fixed reference dataset with MDP off. Corrected five-step qualification passed, but its320-sample shuffle domain differs from the formal1280-sample domain; qualification-prefix parity was not claimed. All six20-step formal comparisons subsequently passed actual320-bin identity. The known16×448 and1×1792 OOM boundaries were not rerun.

## Protected Mantis256: completed real-data four-way

Formal job **760138**, steps 1–4, completed all four real-data arms on the same 16-GPU allocation: TP1/PP2/decoderCP2/EP8/ETP1, MBS1/GBS64/sequence16384, 20 steps/eval0 and primary iterations 4–20 (17 samples). The separately protected Mantis256 slice remains 236 train / 20 validation, unchanged tokenizer/data; this is not mock input or a whole-native dataset claim. Source families remain PR7 `e1484af4` and PR131 `57a5c223`, with reviewed native-reporting/all-rank-memory overlays. PR7 received only the missing Energon geometry flag/provider/producer backport; no loader-serialization or restore-key optimization was added.

| Job.step | Real-data arm | Median step ms | Scheduled tok/s global (per GPU) | PR131 encoder TF/GPU | PR131 decoder TF/GPU | PR131 total TF/GPU | Megatron fixed decoder TF/GPU | Peak allocated / reserved GB |
|---|---|---|---|---|---|---|---|---|
| 760138.1 | PR7 ordinary / MDP off | 22938.0 | 45713.5 (2857.1) | 9.052 | 39.298 | 48.350 | 110.676 | 133.171 / 138.664 |
| 760138.2 | PR7 MDP nonfused | 19071.1 | 54982.5 (3436.4) | 10.900 | 47.319 | 58.220 | 133.288 | 161.015 / 162.785 |
| 760138.3 | PR7 fused-window retain | 9906.3 | 105849.4 (6615.6) | 21.074 | 91.488 | 112.562 | 257.518 | 161.031 / 176.712 |
| 760138.4 | PR131 fused-window retain | 7944.0 | 131996.0 (8249.7) | 26.381 | 114.527 | 140.908 | 322.429 | 181.995 / 183.851 |

**Consumed-work evidence.** All 17 per-step global T/U/R/A tuples are exactly equal across the four arms. Their common primary means are T=671573.176471 content tokens, U=1190615801.294080 squared document lengths, R=1129626.352941 vision patch tokens and A=1255037406.117647 vision attention moment. This establishes matching logged geometry, not identical image/token contents: no exact real-data consumed-sample hash ledger was collected. Full runtime dumps and source/data/tokenizer seals were reviewed; source/environment families still differ between PR7 and PR131.

**Analysis.** Recorded median wall time decreases from ordinary to MDP nonfused to fused execution in this sequential cohort. PR7 fused has a larger reserved-memory peak than PR7 nonfused despite similar allocated peaks; reservation is not live tensor use. PR131 fused has the lowest observed median and higher allocated memory than the PR7 arms. These are observed mode/source trade-offs, not isolated causal gains: one ordered cohort, source/environment differences and absent exact-content identity prevent a replicated winner claim. Encoder/decoder/total rates are native modeled whole-step means, not hardware utilization; memory is the maximum lifetime allocator peak across all 16 ranks, including initialization, not only iterations 4–20. Supplemental 10–20 receipts remain separate.

**Disposition.** All four scheduler steps and the outer job completed 0:0; native/canonical gates, all 16 memory receipts and post-run input seals passed. The user requested direct formal measurement rather than separate new-path qualification. Original data, sources, historical results and the previous Mantis 759796 result remain preserved; the latter is a separate allocation, not an extra replicate of this cohort.

## What changes between mock and real input?

Compare only the **PR131 fused arms**: variable-image mock job 759791 versus protected Mantis256 job 760138. Both use the same PR131 source family/model and native helper, world16/GBS64/MBS1/sequence16384, 20 steps/eval0 and iterations 4–20. Scheduled capacity is 1,048,576 tokens per optimizer step in both. Values below are mean **global logical optimizer-step sums across the workload**, not sums duplicated over CP/PP/GPU replicas. Divide logged mean-per-bin geometry by no extra GPU factor: the native collector multiplies it by GBS once.

| Work per optimizer step | Mock759791 | Real760138 | Real / Mock |
|---|---:|---:|---:|
| Decoder useful-content tokens T | 982,919.65 | 671,573.18 | 0.6832× |
| Encoder pre-merge patch rows R | 1,906,992.24 | 1,129,626.35 | 0.5924× |
| Patch rows per packed training bin R/64 | 29,796.75 | 17,650.41 | 0.5924× |
| Derived merged image-token positions R/4 | 476,748.06 | 282,406.59 | 0.5924× |
| Derived image-token fraction R/(4T) | 48.50% | 42.05% | 0.8670× |
| Decoder forward+backward modeled TFLOP/step | 20,224.40 | 14,406.42 | 0.7123× |
| Encoder forward+backward modeled TFLOP/step | 5,249.87 | 3,318.54 | 0.6321× |
| Encoder+decoder modeled TFLOP/step | 25,474.27 | 17,724.96 | 0.6958× |
| Encoder / decoder work ratio | 25.96% | 23.04% | 0.8874× |
| Encoder share of combined modeled work | 20.61% | 18.72% | 0.9085× |

**Definitions.** TFLOP/step is work, not TFLOPs/GPU/s: apply exact PR131 `training_flops(T,U,R,A)` (FMA=2, conventional training=3×forward), then divide by 1e12, without dividing by step time or GPU count. Fractions are ratios of window totals, not averages of per-step ratios. Spatial merge 2×2 makes the derived repeated image-token count R/4; this excludes vision start/end delimiters and is not an independent token-ID census. T includes image tokens. R/64 is per packed training bin, not per image or raw document. Geometry comes from 12E-formatted logs, not an exact integer-boundary archive.

**Interpretation.** This real slice has less content and fewer total patch rows than this mock workload (Real/Mock 0.6832 and 0.5924), and lower modeled total work (0.6958). However, its mean squared-length moments U and A are larger (1.1756× and 1.0684×); total token/patch counts alone do not describe attention work. These are dataset-shape differences, not measured speedups or GPU-utilization claims. The comparison does not establish that real data is generally smaller, or that encoder CP benefits only long video.

Historical fixed ON job758245 was separately reprocessed through the exact PR131 native helper using its retained T/U/R/A and per-step durations. The six resulting component tuples numerically match the earlier fixed-shape table; this fixed-shape coincidence does not generalize to variable inputs. Original canonical/historical JSON files remain unchanged.

## What changed, and why

| Workstream | Implemented / tested change | Measured question | Current conclusion |
|---|---|---|---|
|Encoder CP and bridge buffers|Contiguous frame partitioning, loader owner/local slicing, group/device/dtype/stream-keyed scratch reuse with owned clones|Does encoder CP2 help versus CP1?|Five ordered pairs completed; CP2 had larger observed step medians. Scratch reuse was enabled in both arms, so its isolated benefit remains unknown|
|MDP and fusion|Ordinary vision → nonfused ownership → fused-window execution → later source revision|How do mode/source changes affect whole-step time?|Four cells completed in one allocation; lower recorded times along the progression, without repeated causal attribution|
|Fixed-input image sweep|Preserved decoder boundaries while changing images per raw sample or image side|How much extra whole-step work does vision introduce?|Six formal cells accepted; 16×448 and 1×1792 OOM during qualification|
|Mantis loader|Shared restore-key correctness repair; candidate alone omits unused JSON/base64 when raw descriptors already exist|Does removing serialization reduce equal-work step time?|36 native tests and model qualifications passed; two orders show small, window-sensitive differences, not a stable winner|
|Whole real-data preparation|Pinned downloads and bounded native PixMo conversion fixtures|Can the requested full native blend be prepared?|Selected transfers/fixtures partly complete; whole-native Mantis/M4/PixMo blend remains unmet|

## Common model, execution and accounting

| Measured source family | Used for | Source boundary |
|---|---|---|
|PR7 `e1484af4f5e9e5723105f731fb555dba9a32fecb` lineage|Ordinary/nonfused/fused comparison arms|Original per-job source and normalization receipts apply|
|Older PR131 `57a5c2239242340cad5c22a8dd3fec18b16015e9` lineage|Encoder CP / bridge-buffer pairs; PR131 four-way arm and original Mantis|CP/buffer overlays and native grid/NVTX compatibility overlays are distinct, not one shared source tree|
|PR131 `bbba1001543a9c99a26ae8a8370b40e005c80d80` lineage|New fixed sweep and Mantis loader ablation|Separate fixed-receipt/grid compatibility and shared restore-key / candidate serialization overlays retain their own seals|

The publication branch is documentation, not the measured source tree or a claim that these implementation overlays have merged.

### Model and hardware cohorts

| Cohort | Model identity / hardware | Execution boundary |
|---|---|---|
|Current GB300 VLM measurements|PR7/PR131 hybrid VLM, configured `model_arch=qwen3vl`; 48 decoder layers, hidden 2048, 128 experts; 27 vision layers; padded vocabulary 248448, KV channels 128|16 GB300 GPUs. Not an unqualified claim of official canonical Qwen3-VL equivalence|
|Archived text anchors|Five `qwen3_30b_a3b` and six later `qwen3` artifacts|Recorded world64/GBS512; historical GB200 campaign context, not new per-attempt hardware proof|
|Historical VL/hybrid catalog|22 archived `qwen35_vl`, 175 hybrid-labelled, 19 current `qwen3vl`, four model-unknown artifacts|Different model/source/data cohorts; no cross-cohort speed ranking. Five recorded 128/256 topology products remain unvalidated|

Current 16-GPU common parallelism is TP1/PP2/decoder CP2/DP4/EP8/ETP1, MBS1/GBS64, sequence 16384, BF16 and random initialization without checkpoint loading. Source/runtime receipts belong to each job. The new fixed sweep and loader pairs use HybridEP (32 SMs, chunks 128), distributed optimizer, fused-window **retain** with cap 131072, no CUDA graphs/recompute/MTP, effective gradient/parameter overlap off, and effective sequence parallel false at TP1. Encoder CP is 1 unless explicitly varied. Earlier four-way mode/source and encoder-CP cells retain their original stack; the new settings are not retroactively assigned to them.

| Quantity | Definition / publication rule |
|---|---|
|Step time|Median measured whole-VLM optimizer-step wall time; primary performance fact|
|Scheduled global tok/s|`GBS × configured sequence length × 1000 / median step ms`; capacity-normalized at the reported median, not actual useful/content tokens, not a mean of stepwise rates, and no GPU-count multiplier|
|Per-GPU tok/s / component units|Parentheses divide the global rate by 16 only for independently validated current 16-GPU runs. Historical configuration-only world values yield per-GPU N/A. TF/GPU means TFLOPs/GPU/s; encoder, decoder and total columns share whole-step time, not encoder-only elapsed time|
|Current PR131-native encoder / decoder / total TF|The new reruns use the exact pinned PR131 useful-content helper and collector for all three components. Rates exclude padding and non-matmul work (including encoder attention head-width padding72→128), use whole-step time, and are not hardware FLOPs. Raw Megatron fixed decoder remains a separate column|
|Historical fixed-cell accounting|The six older fixed cells retain their originally published attended-padded decoder proof and encoder estimate. Exact PR131-native reaggregation independently gives the same numerical component tuples for these fixed inputs; that coincidence does not change variable-input semantics or overwrite historical JSON|
|Window|New 20-step runs: primary 4–20 (17 samples), supplemental 10–20 (11). Older four-way: 10–50 (41). Older CP table: supplemental 10–20. Never combine these into one percentage|
|Corrected native decoder TFLOPs/GPU/s|Sealed native decoder formula with resolved arguments and verified final padded boundaries, divided by original whole-step time and world size; available for the six new fixed cells only|
|Mean versus pooled corrected rate|Mean averages stepwise rates; pooled divides total modeled work by total elapsed time. Neither is a step-time median|
|Legacy / useful-content estimates|Different numerators; preserved in [machine-readable historical audit](per-experiment.json), not relabelled as corrected native rates|
|Unavailable / null|Missing proof, not zero work. Historical/nonstatic final-boundary corrections and historical all-rank peaks remain unavailable where not captured. New rerun native content rates and complete lifetime allocator peaks are reported separately above|
|Interpretation|Decoder-only modeled work excludes varying vision work; not hardware counters, utilization, MFU or convergence evidence|

## Completed mock measurements: MDP / fused-window / source

The [new 20-step reruns above](#new-20-step-reruns-pr131-native-components-and-all-rank-memory) add exact PR131-native component rates and all-rank allocator peaks; the older measurement history below is retained unchanged.

One physical allocation, job **752159**, with native variable mock inputs, world16/GBS64/MBS1 and the decoder topology above. These four rows share the original 50-step protocol. Mode/source and PR-specific normalization environment differences are the declared axes. All four measurement cells completed; the outer job later reached TIMEOUT during a separate packing-profile startup. Measurement completion is not a clean outer-job claim.

| Job | Source / vision mode | Status / total steps | Window | Median step ms | Scheduled tok/s global (per GPU) | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 752159 | PR7 / ordinary, MDP off | Completed / 50 | 10–50 | 13804.5 | 75959.0 (4747.4) | N/A | unavailable | N/A |
| 752159 | PR7 / MDP nonfused | Completed / 50 | 10–50 | 11479.4 | 91344.1 (5709.0) | N/A | unavailable | N/A |
| 752159 | PR7 / fused-window retain | Completed / 50 | 10–50 | 9820.5 | 106774.2 (6673.4) | N/A | unavailable | N/A |
| 752159 | PR131 / fused-window retain | Completed / 50 | 10–50 | 8901.4 | 117799.0 (7362.4) | N/A | unavailable | N/A |

**Analysis.** The lower whole-step times motivate fusion/ownership investigation, but do not isolate one function or encoder-only savings. Exact historical consumed geometry is not archived; generator recipe equality is not input-identity proof. No repeated/randomized variance estimate exists. Standalone baseline 751915 (14961 ms) is separate and must not replace the ordinary row.

## Encoder CP and buffer reuse: five completed pairs

Older owned source implements contiguous real-frame partitioning, loader owner/local slicing and stock TE full-attention padding; decoder zigzag behavior is unchanged. Output/gradient and full 27-vision/48-decoder model tests passed. Scratch reuse is **on in both arms**, with attention/router/preprocess graph scopes and graph warmup2. This differs from the graph-free new sweep and loader stack. CPU full-image materialization still occurs before local slicing. This code is not automatically ported to latest `bbba100`.

| Paired job | Workload / intended axis | Status | Window | Encoder CP1 median ms | Encoder CP2 median ms | Scheduled tok/s global (per GPU): CP1 / CP2 | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 749724 | Variable reference / encoder CP | Both completed | 10–20 | 4358.8 | 4538.4 | 240565.3 (15035.3) / 231045.3 (14440.3) | N/A | unavailable | N/A |
| 750008 | Fixed image area 1× / encoder CP | Both completed | 10–20 | 4144.3 | 4293.5 | 253016.4 (15813.5) / 244224.1 (15264.0) | N/A | unavailable | N/A |
| 750020 | Fixed image area 2× / encoder CP | Both completed | 10–20 | 4151.5 | 4267.2 | 252577.6 (15786.1) / 245729.3 (15358.1) | N/A | unavailable | N/A |
| 750022 | Fixed image area 4× / encoder CP | Both completed | 10–20 | 4574.4 | 4806.8 | 229227.0 (14326.7) / 218144.3 (13634.0) | N/A | unavailable | N/A |
| 750023 | Fixed image area 8× / encoder CP | Both completed | 10–20 | 5571.4 | 6054.7 | 188206.9 (11762.9) / 173183.8 (10824.0) | N/A | unavailable | N/A |

**Analysis.** CP2 had larger observed medians in every pair. Each job is one ordered comparison on the same allocation; different image-area jobs are not identical workloads. Increasing encoder CP changes ownership within the fixed inner-DP group, not simply half of every rank's work. The area multipliers above are not the new image-count/side-length axis below. No scratch-only ablation, whole-patch versus unmodified-source comparison, or repeated ≥5% encoder-CP win was established. Next separate owner-only CPU materialization and scratch on/off, with exact pixel/order/gradient parity before timing.

## Fixed-input image count / size: six accepted cells and two OOM boundaries

Job **758245**, unchanged sealed PR131-derived stack, 20 iterations/eval0, LR warmup/decay 2/20. Each raw sample has a 4096-token document; four documents fill each 16384-token bin. Image counts are **per raw sample**, so each packed bin has four times the table count. Static THD has 32 slots (33 endpoints), four real segments and repeated terminal endpoints.

| Job / step | Images × side | Status | Window | Median step ms | Scheduled tok/s global (per GPU) | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean | Corrected decoder TF/GPU pooled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 758245.1 | 1 × 224 | Formal accepted | 4–20 | 6475.1 | 161939.7 (10121.2) | 1.2528 | 243.9546 | 245.2073 | 243.6629 |
| 758245.3 | 1 × 448 | Formal accepted | 4–20 | 6586.1 | 159210.5 (9950.7) | 5.3647 | 240.8134 | 246.1781 | 240.4518 |
| 758245.5 | 1 × 896 | Formal accepted | 4–20 | 7245.8 | 144715.0 (9044.7) | 25.3971 | 217.2698 | 242.6669 | 217.0908 |
| 758245.7 | 2 × 448 | Formal accepted | 4–20 | 6702.8 | 156438.5 (9777.4) | 10.5125 | 235.9466 | 246.4591 | 235.6871 |
| 758245.9 | 4 × 448 | Formal accepted | 4–20 | 6955.6 | 150752.8 (9422.0) | 20.1315 | 225.9198 | 246.0513 | 225.6909 |
| 758245.11 | 8 × 448 | Formal accepted | 4–20 | 7966.6 | 131621.5 (8226.3) | 35.3368 | 198.2781 | 233.6148 | 198.2379 |
| 753812.4 | 16 × 448 | Qualification OOM | No accepted window | unavailable | N/A | N/A | unavailable | N/A | unavailable |
| 758212.10 | 1 × 1792 | Qualification OOM | No accepted window | unavailable | N/A | N/A | unavailable | N/A | unavailable |

Global decoder moments are constant: Tpad=1,048,576 and Upad=4,294,967,296 per step. **Encoder is a useful-content matmul model; decoder is the accepted padded-native model; E+D is their mixed modeled sum**, computed before rounding over the same 17 steps and world16 denominator. The encoder omits actual 72→128 attention head-width padding, non-matmul operations and exact backward instruction differences; none of these columns measures hardware utilization or encoder-only throughput. The [six-cell snapshot](sweep-corrected-snapshot-20260916.md) gives the encoder formula/source hashes. Original [decoder JSON](sweep-corrected-snapshot-20260916.json) and its supplemental windows remain unchanged.

**Analysis.** Extra vision work increases whole-step time and lowers the constant-work decoder rate; that is not falling hardware utilization or an optimization ranking. Four 448 images and one 896 image have equal raw-patch sum R=802816, but attention moment A=629407744 versus 2517630976 (4×); observed steps are 6955.6 versus 7245.8 ms. Eight 448 images have more patches (R=1605632) but lower A=1258815488 than one 896 image, at 7966.6 ms. Patch-linear and per-image quadratic work both matter; this is consistent with attention cost, not isolated causal proof. OOM cells were not silently downscaled into successful replacements.

## Real Mantis: protected slice, not a full blend

The original protected slice now has a [new 20-step native-accounting/memory rerun](#new-20-step-reruns-pr131-native-components-and-all-rank-memory), separate from the historical Mantis and loader-ablation results below.

Same current model/world16/GBS64/MBS1, but real conversations/images are a different workload from mock. The original Mantis256 slice remains separately protected: **236 train / 20 validation**, eval0. Native image bounds are 200704–1003520 pixels, workers0, packing buffer16 and `max_samples_per_sequence=4`; this knob is not asserted to be a universal four-document cap. Nonstatic attention differs from the fixed mock layout.

| Job / source | Dataset | Status | Window | Median step ms | Scheduled tok/s global (per GPU) | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 752807 / original eager path | Protected Mantis256 | Formal completed | 4–20 | 7221.1 | 145210.0 (9075.6) | N/A | unavailable | N/A |
| 752807 / same run, supplemental | Same slice | Not a second experiment | 10–20 | 7221.1 | 145210.0 (9075.6) | N/A | unavailable | N/A |

| Distribution / preparation evidence | Observed result | Limit |
|---|---|---|
|Mantis training tokens|Median 838; p90 2520; maximum 3221|Prepared rows, not fixed mock tokens|
|Mantis images per training row|1/2/3/4 images: 126/31/37/42 rows|236 training records; not images per consumed packed bin|
|Mantis consumed content|Mean about 10493 tokens per 16384-budget pack (~64%)|Content occupancy, not TE compute utilization|
|Nemotron preparation|Prepared token median 6335|CPU preparation/packing only, no model result|
|Selected source downloads|Mantis 35/36; M4 40/41; PixMo 111/111; backing frames 17/17 files|Backing set is 16 TAR archives + README; partial downloads preserved|
|PixMo native fixtures|32, 512 and 4096 rows passed; 4096 included / zero excluded with hashes and image/conversation parity|Not full-corpus preparation or training|
|Whole-native 1:1:1 blend|Unmet; M4 temporal mapping unqualified|Full preparation deferred; approximate 118–132 min estimate exceeded remaining budget|

**Analysis.** Static GBS×sequence estimates over-credit short content when substituted for actual workload accounting. Mantis speed is not a mock loader-speed comparison. Small-corpus repetition and packing change effective work; neither prepared row counts nor recipe equality prove historical consumed identity.

## Dataloader ablation: eager descriptors versus omitted unused JSON

Both new variants share the independently attributed restore-key correctness repair. Only the candidate omits JSON/base64 serialization when the raw-descriptor handoff exists; fallback/empty paths remain covered. V5 passed **36 native tests**, including all four DP streams, save-two/restore-two and four-owner pixel checks; both variants passed ten-step model qualification. Formal pairs use the same qualified source/runtime/data/stack within each job; all logged arguments except output directory and all twenty T/U/R/A moments agree. Historical 752807 is **not** the corrected baseline.

| Job / order | Status | Window / samples | Eager baseline median ms | Candidate median ms | Observed reduction | Scheduled tok/s global (per GPU): baseline / candidate | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 758425 / baseline → candidate | Clean formal pair | 4–20 / 17 | 7498.3 | 7306.2 | 2.5619% | 139841.8 (8740.1) / 143518.7 (8969.9) | N/A | unavailable | N/A |
| 758425 / same pair | Supplemental | 10–20 / 11 | 7318.8 | 7306.2 | 0.1722% | 143271.6 (8954.5) / 143518.7 (8969.9) | N/A | unavailable | N/A |
| 758592 / candidate → baseline | Recovered model measurements; outer failed | 4–20 / 17 | 8977.5 | 8845.0 | 1.4759% | 116800.4 (7300.0) / 118550.1 (7409.4) | N/A | unavailable | N/A |
| 758592 / same pair | Supplemental | 10–20 / 11 | 9004.5 | 8776.7 | 2.5298% | 116450.2 (7278.1) / 119472.7 (7467.0) | N/A | unavailable | N/A |

**Failure recovery.** Job 758592 retains outer/batch `FAILED 1:0`; both training steps completed `0:0`. The reverse wrapper ended with a baseline source alias then checked it against the candidate manifest. A separate correct-root read-only check passed all 2527 baseline files, 2530 candidate files, prepared data and tokenizer; original before-check evidence passed. Individual evaluators plus independent order/log/argument/geometry/window review accepted only the scoped model measurements. No source/model edit or training rerun erased the failed receipt.

**Analysis.** Candidate medians are smaller in both orders, but the 0.17–2.56% differences depend on the window. One pair per order on different racks/allocations does not establish variance, a stable winner or causal I/O speedup. Do not pool raw times across jobs. Bounded parity and equal moments do not archive every consumed token/image or final nonstatic attended boundary; corrected rates and peak memory stay unavailable. Quiescent restore is not live-prefetch or training-checkpoint resume proof.

## Historical campaign panels: recorded configuration evidence

These anchors are retained observations, not newly reconstructed source/runtime parity. Their detailed original windows and all 231 rows remain in the appendices and [recorded per-experiment details](per-experiment.md). Corrected rates are unavailable throughout this panel. Do not compare recorded world64 historical GB200 context with the new world16 GB300 measurements.

| Namespace / recorded contrast | Common recorded config | Variable | Median step A → B ms | Scheduled tok/s global (per GPU): listed order | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-phase4-final EXP-040 → 041 (artifacts 001/002) | Text, TP1/PP1/EP8, GBS512, sequence4096 | Decoder CP1 → 2 | 4122.6 → 10899.2 | 508696.5 (N/A) / 192413.4 (N/A) | N/A | unavailable | N/A |
| qwen3-phase4-final EXP-042 → 043 (003/004) | Text, CP1/EP8, GBS512 | Sequence8192 → 16384 | 4602.4 → 8023.3 | 911329.7 (N/A) / 1045530.9 (N/A) | N/A | unavailable | N/A |
| active-reset-20260429 EXP-000 → 016 (028/040) | Hybrid, TP1/PP1/CP1, GBS512/MBS1, sequence4096, HybridEP, image224 | EP8 → 32 only in recorded YAML | 5571.7 → 4798.6 | 376393.6 (N/A) / 437034.1 (N/A) | N/A | unavailable | N/A |
| active-reset-20260429 EXP-019 → 025 (043/047) | Hybrid, CP1/EP32, GBS512, sequence16384, image224 | FP8 off → hybrid / mxfp8 | 12071.7 → 11922.6 | 694898.6 (N/A) / 703588.8 (N/A) | N/A | unavailable | N/A |
| active-reset-20260429 EXP-039 → 044 (055/060) | Hybrid, CP2/EP32, GBS512, sequence16384, THD enabled, image224 | FP8 off → hybrid / mxfp8 | 17831.3 → 15783.4 | 470442.9 (N/A) / 531483.0 (N/A) | N/A | unavailable | N/A |
| Archived qwen35_vl EXP-027/028/029 (023/024/025) | CP1/EP16, sequence16384 | Pack2 / 4 / 8 | 13610.3 / 13522.8 / 14057.6 | 616342.6 (N/A) / 620330.7 (N/A) / 596731.2 (N/A) | N/A | unavailable | N/A |

**Analysis.** EP effects are nonmonotonic: related active EXP-014/015/017 (EP4/16/64) record 6459.3/5039.7/8235.7 ms. FP8 differences are small in the CP1 anchor and larger in CP2/THD; CP/DP and THD differ across those anchors, so there is no universal FP8 gain. Text CP execution does not imply short-sequence efficiency. Longer sequence changes work, not just throughput. Graph/overlap/dispatcher/recompute changes outside these specific contrasts remain separate axes, not silently controlled variables.

Catalog uncertainty is preserved: 160 providers unspecified, 69 real-labelled and two explicitly mock; unspecified is not mock. The active namespace has 108 HybridEP, 90 all-to-all and six unspecified dispatchers. Four model-unknown artifacts are failures; two partial artifacts have known hybrid labels. The 141 unlinked experiment directories are unknown, not failures. The category index assigns all 231 artifacts once, with explicit cross-references rather than invented experiment intent.

## Completed profiles: diagnostic evidence, not throughput benchmarks

All four cells have finalized SQLite integrity/stability and **16 GPU-worker** coverage. Ordinary/nonfused used 752159; fused continuations used 753568/753569, different racks and LR/eval 2/20/0 instead of 5/50/default. Capture 5–8 maps to displayed 6–8 only by source inference: no explicit iteration anchors.

| Profile job / cell | Status | Worker kernel span s | Selected vision ranges/node | Forward-bridge kernel sums by node 0–3, s | Scheduled tok/s global (per GPU) | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 752159 / ordinary PR7 | Finalized, integrity PASS | 64.43–64.60 | Named outer range absent; vision executes | not applicable | N/A | N/A | N/A | N/A |
| 752159 / nonfused PR7 | Finalized, integrity PASS | 66.83–66.98 | 192 | 2.870 / 20.881 / 9.467 / 5.976 | N/A | N/A | N/A | N/A |
| 753568 / fused PR7 | Finalized, integrity PASS | 40.10–40.23 | 48 | 0.319 / 0.466 / 1.785 / 0.715 | N/A | N/A | N/A | N/A |
| 753569 / fused PR131 | Finalized, integrity PASS | 47.98–48.10 | 48 | 0.738 / 0.885 / 2.476 / 4.096 | N/A | N/A | N/A | N/A |

**Analysis.** Process-aware CUDA correlations matched 4,220,973 packing and 4,277,998 latest kernels. HybridEP/NCCL and synchronization tails are observed, but event sums overlap: no wall-time fractions, global utilization or cross-node critical path follow. A waiting rank is not necessarily causal. The 192→48 range count is fusion granularity, not four times less image work. Interpolation-associated GPU sums are small relative to CPU envelopes; remaining host time includes unattributed work/profiler overhead. [Detailed profile report](profile-summary.md) retains denominators and source/capture caveats.

## Failure disposition and remaining gates

| Attempt / condition | Disposition | What is and is not established |
|---|---|---|
|753812.4 / 16 images × 448|Qualification CUDA OOM|No formal metric for this shape|
|758212.10 / 1 image × 1792|Qualification CUDA OOM|No formal metric for this shape|
|754294 / original native loader|Eight synthetic tests passed; first real DP0 restore failed|Original missing-key bug preserved; no model launched in that attempt|
|758212 / v4 loader probe|Ten tests passed; tuple/list ownership assertion failed|Test inspector corrected separately; exact fingerprint/FIFO gates retained in v5|
|758592 / reverse final source check|Outer failed; independent correct-root recovery accepted model measurements|Both training steps completed; not a clean outer job|
|Mantis/M4 whole source transfer|35/36 and 40/41 selected files complete|Partials retained; no full-native/blend claim|
|Historical catalog gaps|Missing source/runtime/input/boundary proof and five topology ambiguities|No corrected rates invented; unlinked directories are not presumed failed|

Priorities are concrete gates, not proven bottlenecks: repeat matched loader pairs; test encoder owner-only CPU materialization with exact pixel/order/gradient parity; isolate scratch reuse on/off; add explicit step anchors and process-aware communication ownership. Whole-native preparation and temporal semantics remain required before a full real-data blend. The requested repeated ≥5% encoder-CP win is **not achieved**.

## Reproducibility and review

The [six-cell JSON](sweep-corrected-snapshot-20260916.json) and [evidence fingerprints](evidence-fingerprints.md) preserve accepted accounting and provenance. [Data/configuration details](data-and-configs.md), [progress/failures](24h-progress.md), [historical catalog](campaign-wide-catalog.md) and [per-experiment JSON](per-experiment.json) retain scope and unavailable fields. [Accounting utilities](accounting-utilities.md) have seven standard-library tests and native formula equivalence checks; historical replay remains unexecuted.

No raw logs, private paths, media, credentials or multi-gigabyte traces are bundled. Independent primary reviews passed; the external second-model service was unavailable, so no external agreement is claimed. The campaign is not fully complete. The following unchanged appendices preserve historical timings, category membership and every historical record.

<details>
<summary>Historical timing, pairing and correction audit notes</summary>

## Historical timing audit

Each row below comes from an accepted legacy result artifact. Step time is the median in the stated window; native TFLOPs is the legacy verifier arithmetic mean, not a hardware measurement or median. The old raw label `iters 10-50` sometimes describes only the filter: a20-step run has actual10–20 coverage (11 samples), while10–50 has41 samples. These are separate experiments unless explicitly paired.

| Result identifier | Actual window | GBS | Step median ms | Legacy native TFLOPs/GPU mean | Scheduled tok/s global (per GPU) | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-PR131-ENCODER-CP1-GB300 | 10–20 | 64 | 4170.2 | 601.9 | 251445.0 (15715.3) | N/A | null | N/A |
| EXP-PR131-ENCODER-CP1-PAIR749724-GB300 | 10–20 | 64 | 4358.8 | 578.1364 | 240565.3 (15035.3) | N/A | null | N/A |
| EXP-PR131-ENCODER-CP2-PAIR749724-GB300 | 10–20 | 64 | 4538.4 | 551.1273 | 231045.3 (14440.3) | N/A | null | N/A |
| EXP-PR131-IMAGE1X-CP1-GB300 | 10–20 | 64 | 4144.3 | 603.5182 | 253016.4 (15813.5) | N/A | null | N/A |
| EXP-PR131-IMAGE1X-CP2-GB300 | 10–20 | 64 | 4293.5 | 583.2 | 244224.1 (15264.0) | N/A | null | N/A |
| EXP-PR131-IMAGE2X-CP1-GB300 | 10–20 | 64 | 4151.5 | 594.5636 | 252577.6 (15786.1) | N/A | null | N/A |
| EXP-PR131-IMAGE2X-CP2-GB300 | 10–20 | 64 | 4267.2 | 578.5545 | 245729.3 (15358.1) | N/A | null | N/A |
| EXP-PR131-IMAGE4X-CP1-GB300 | 10–20 | 64 | 4574.4 | 542.6 | 229227.0 (14326.7) | N/A | null | N/A |
| EXP-PR131-IMAGE4X-CP2-GB300 | 10–20 | 64 | 4806.8 | 512.7182 | 218144.3 (13634.0) | N/A | null | N/A |
| EXP-PR131-IMAGE8X-CP1-GB300 | 10–20 | 64 | 5571.4 | 452.8364 | 188206.9 (11762.9) | N/A | null | N/A |
| EXP-PR131-IMAGE8X-CP2-GB300 | 10–20 | 64 | 6054.7 | 414.6909 | 173183.8 (10824.0) | N/A | null | N/A |
| EXP-PR131-MANTIS16-752807-20-GB300-legacy-window | 10–20 | 64 | 7221.1 | 341.0182 | 145210.0 (9075.6) | N/A | null | N/A |
| EXP-PR131-TFLOPS16-SEQ4096-GBS256 | 10–50 | 256 | 7296 | 217.3512 | 143719.3 (8982.5) | N/A | null | N/A |
| EXP-PR7-FOURWAY-751915-pr7_baseline | 10–50 | 64 | 14961 | 162.1098 | 70087.3 (4380.5) | N/A | null | N/A |
| EXP-PR7-FOURWAY-752159-pr131_latest | 10–50 | 64 | 8901.4 | 285.1317 | 117799.0 (7362.4) | N/A | null | N/A |
| EXP-PR7-FOURWAY-752159-pr7_baseline | 10–50 | 64 | 13804.5 | 176.2098 | 75959.0 (4747.4) | N/A | null | N/A |
| EXP-PR7-FOURWAY-752159-pr7_mdp | 10–50 | 64 | 11479.4 | 209.8732 | 91344.1 (5709.0) | N/A | null | N/A |
| EXP-PR7-FOURWAY-752159-pr7_packing | 10–50 | 64 | 9820.5 | 241.1366 | 106774.2 (6673.4) | N/A | null | N/A |

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


</details>

## Historical category and cohort index

This primary navigation partition covers all 231 artifacts exactly once; analytical topics can overlap and are cross-referenced in the main sections. Membership reflects recorded fields, not inferred experiment-number ranges. Nine EP/precision anchor records are assigned to B from independently inspected YAML contrasts (artifact-028/038/039/040/041/043/047/055/060), overriding generic image geometry. Remaining priority: failed/partial status; explicit text label; explicit real/Energon/Mantis provider; encoder-CP with graph scope; MDP mode without graph scope; explicit image geometry at decoder CP1 without recorded FP8/recompute; otherwise sequence/stack configurations. The latter are subdivided by recorded CP, sequence and FP8, but their specific optimization intent remains unresolved. Provider-unspecified records remain unspecified; empty fields are not proof of defaults. Hardware/model cohorts and unknowns stay separate.

<details>
<summary>A. Text-model CP and sequence length — 11 artifacts</summary>

| Recorded cohort / configuration | Artifact IDs and original experiment IDs |
|---|---|
| qwen3-phase4-final; qwen3_30b_a3b; world64 recorded; GB200 campaign context only; GBS512 | artifact-001 (EXP-040), artifact-002 (EXP-041), artifact-003 (EXP-042), artifact-004 (EXP-043), artifact-005 (EXP-045) |
| active-reset-20260429; qwen3; world64 recorded; GB200 campaign context only; GBS512 | artifact-088 (EXP-070), artifact-089 (EXP-071v3), artifact-090 (EXP-072), artifact-091 (EXP-073v4), artifact-093 (EXP-075v2), artifact-094 (EXP-076v2) |

</details>

<details>
<summary>B. Parallelism/precision anchors and remaining sequence-stack configurations — 103 artifacts</summary>

| Recorded cohort / configuration | Artifact IDs and original experiment IDs |
|---|---|
| qwen35vl-phase0-3; qwen35_vl_35b_a3b; world64 recorded; GB200 campaign context only; GBS512; CP1, seq4096, FP8=off | artifact-006 (EXP-001) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP1, seq4096, FP8=off; reviewed EP/precision contrast | artifact-028 (EXP-000), artifact-038 (EXP-014), artifact-039 (EXP-015), artifact-040 (EXP-016), artifact-041 (EXP-017) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq4096, FP8=off | artifact-033 (EXP-009), artifact-034 (EXP-010), artifact-044 (EXP-020), artifact-045 (EXP-022), artifact-046 (EXP-023), artifact-049 (EXP-030), artifact-050 (EXP-034), artifact-051 (EXP-035), artifact-052 (EXP-036), artifact-053 (EXP-037), artifact-054 (EXP-038), artifact-074 (EXP-054final2), artifact-075 (EXP-055fix), artifact-076 (EXP-056fix), artifact-081 (EXP-059fix), artifact-083 (EXP-060fix), artifact-204 (EXP-bridge1), artifact-205 (EXP-bridge2), artifact-209 (EXP-va4), artifact-212 (EXP-varimg2) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP1, seq16384, FP8=off; reviewed EP/precision contrast | artifact-043 (EXP-019) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP1, seq16384, FP8=hybrid; reviewed EP/precision contrast | artifact-047 (EXP-025) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP1, seq32768, FP8=hybrid | artifact-048 (EXP-026) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq16384, FP8=off; reviewed EP/precision contrast | artifact-055 (EXP-039) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq8192, FP8=off | artifact-056 (EXP-040fix), artifact-068 (EXP-052v4), artifact-069 (EXP-052v5), artifact-070 (EXP-052v6), artifact-071 (EXP-052v7), artifact-077 (EXP-057), artifact-078 (EXP-057fix), artifact-086 (EXP-064), artifact-092 (EXP-074), artifact-096 (EXP-078v7), artifact-097 (EXP-079v4), artifact-098 (EXP-080v2), artifact-099 (EXP-080v3), artifact-100 (EXP-083), artifact-101 (EXP-084), artifact-102 (EXP-085), artifact-103 (EXP-090), artifact-104 (EXP-092), artifact-105 (EXP-093), artifact-106 (EXP-094), artifact-107 (EXP-095-mock), artifact-108 (EXP-096-mock), artifact-110 (EXP-107), artifact-114 (EXP-115), artifact-115 (EXP-116), artifact-116 (EXP-117), artifact-119 (EXP-121), artifact-120 (EXP-123), artifact-121 (EXP-124), artifact-126 (EXP-135), artifact-223 (EXP-vc4) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq8192, FP8=hybrid | artifact-057 (EXP-041), artifact-064 (EXP-048) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq32768, FP8=hybrid | artifact-058 (EXP-042), artifact-073 (EXP-054), artifact-080 (EXP-059), artifact-082 (EXP-060), artifact-084 (EXP-061), artifact-087 (EXP-065) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq32768, FP8=off | artifact-059 (EXP-043), artifact-061 (EXP-045), artifact-217 (EXP-vb3), artifact-219 (EXP-vb5) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq16384, FP8=hybrid; reviewed EP/precision contrast | artifact-060 (EXP-044) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP4, seq16384, FP8=hybrid | artifact-062 (EXP-046) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP4, seq32768, FP8=hybrid | artifact-063 (EXP-047) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP4, seq65536, FP8=hybrid | artifact-065 (EXP-050) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP4, seq65536, FP8=off | artifact-067 (EXP-052) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq16384, FP8=hybrid | artifact-072 (EXP-053), artifact-079 (EXP-058), artifact-085 (EXP-062) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP4, seq8192, FP8=off | artifact-109 (EXP-099) |
| active-reset-20260429; qwen3vl; 16GB300 current-thread cohort; GBS256; CP1, seq4096, FP8=unspecified | artifact-198 (EXP-PR131-TFLOPS16-SEQ4096-GBS256) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq12288, FP8=off | artifact-206 (EXP-va1), artifact-207 (EXP-va2), artifact-208 (EXP-va3), artifact-210 (EXP-va5), artifact-211 (EXP-va6), artifact-214 (EXP-varlen6), artifact-221 (EXP-vc1), artifact-222 (EXP-vc3), artifact-224 (EXP-vc5) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq16384, FP8=off | artifact-215 (EXP-vb1), artifact-218 (EXP-vb4), artifact-220 (EXP-vb6) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq24576, FP8=off | artifact-216 (EXP-vb2), artifact-225 (EXP-vc6) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq12288, FP8=unspecified | artifact-226 (EXP-vd1), artifact-227 (EXP-vd2), artifact-230 (EXP-vd5) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq32768, FP8=unspecified | artifact-228 (EXP-vd3), artifact-229 (EXP-vd4) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512; CP2, seq16384, FP8=unspecified | artifact-231 (EXP-vd6) |

</details>

<details>
<summary>C. Explicit encoder-CP with graph scope — 11 artifacts</summary>

| Recorded cohort / configuration | Artifact IDs and original experiment IDs |
|---|---|
| active-reset-20260429; qwen3vl; 16GB300 current-thread cohort; GBS64 | artifact-185 (EXP-PR131-ENCODER-CP1-GB300), artifact-186 (EXP-PR131-ENCODER-CP1-PAIR749724-GB300), artifact-187 (EXP-PR131-ENCODER-CP2-PAIR749724-GB300), artifact-188 (EXP-PR131-IMAGE1X-CP1-GB300), artifact-189 (EXP-PR131-IMAGE1X-CP2-GB300), artifact-190 (EXP-PR131-IMAGE2X-CP1-GB300), artifact-191 (EXP-PR131-IMAGE2X-CP2-GB300), artifact-192 (EXP-PR131-IMAGE4X-CP1-GB300), artifact-193 (EXP-PR131-IMAGE4X-CP2-GB300), artifact-194 (EXP-PR131-IMAGE8X-CP1-GB300), artifact-195 (EXP-PR131-IMAGE8X-CP2-GB300) |

</details>

<details>
<summary>D. Explicit image geometry at decoder CP1 — 32 artifacts</summary>

| Recorded cohort / configuration | Artifact IDs and original experiment IDs |
|---|---|
| qwen35vl-phase0-3; qwen35_vl_35b_a3b; world64 recorded; GB200 campaign context only; GBS512 | artifact-007 (EXP-002), artifact-008 (EXP-003), artifact-009 (EXP-004), artifact-010 (EXP-005), artifact-011 (EXP-006), artifact-012 (EXP-007), artifact-013 (EXP-008), artifact-014 (EXP-009), artifact-015 (EXP-010), artifact-016 (EXP-011), artifact-017 (EXP-012), artifact-018 (EXP-013), artifact-019 (EXP-014), artifact-020 (EXP-015), artifact-021 (EXP-017), artifact-022 (EXP-019), artifact-023 (EXP-027), artifact-024 (EXP-028), artifact-025 (EXP-029), artifact-026 (EXP-030), artifact-027 (EXP-031) |
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512 | artifact-029 (EXP-003), artifact-030 (EXP-004), artifact-031 (EXP-005), artifact-032 (EXP-006), artifact-035 (EXP-011), artifact-036 (EXP-012), artifact-037 (EXP-013), artifact-042 (EXP-018), artifact-066 (EXP-051), artifact-095 (EXP-077v8), artifact-213 (EXP-varlen3) |

</details>

<details>
<summary>E. Explicit MDP mode without graph scope — 5 artifacts</summary>

| Recorded cohort / configuration | Artifact IDs and original experiment IDs |
|---|---|
| active-reset-20260429; qwen3vl; 16GB300 current-thread cohort; GBS64 | artifact-199 (EXP-PR7-FOURWAY-751915-pr7_baseline), artifact-200 (EXP-PR7-FOURWAY-752159-pr131_latest), artifact-201 (EXP-PR7-FOURWAY-752159-pr7_baseline), artifact-202 (EXP-PR7-FOURWAY-752159-pr7_mdp), artifact-203 (EXP-PR7-FOURWAY-752159-pr7_packing) |

</details>

<details>
<summary>F. Explicit real-data provider — 63 artifacts</summary>

| Recorded cohort / configuration | Artifact IDs and original experiment IDs |
|---|---|
| active-reset-20260429; qwen3vl_hybrid; world64 recorded; GB200 campaign context only; GBS512 | artifact-113 (EXP-110), artifact-117 (EXP-118), artifact-118 (EXP-119), artifact-122 (EXP-125), artifact-123 (EXP-126), artifact-124 (EXP-133), artifact-125 (EXP-134), artifact-127 (EXP-136), artifact-128 (EXP-137), artifact-129 (EXP-138), artifact-130 (EXP-140), artifact-131 (EXP-141), artifact-132 (EXP-142), artifact-133 (EXP-143), artifact-134 (EXP-152), artifact-135 (EXP-153), artifact-136 (EXP-154), artifact-137 (EXP-160), artifact-143 (EXP-172), artifact-144 (EXP-180), artifact-145 (EXP-182), artifact-146 (EXP-185), artifact-147 (EXP-200), artifact-148 (UNKNOWN), artifact-149 (UNKNOWN), artifact-150 (UNKNOWN), artifact-153 (UNKNOWN), artifact-154 (UNKNOWN), artifact-157 (EXP-230), artifact-158 (EXP-231), artifact-159 (EXP-232), artifact-160 (EXP-233), artifact-161 (EXP-234), artifact-162 (EXP-235), artifact-163 (EXP-236), artifact-164 (EXP-237), artifact-165 (EXP-238), artifact-166 (EXP-239), artifact-167 (EXP-240), artifact-168 (EXP-241), artifact-169 (EXP-246), artifact-170 (EXP-247), artifact-171 (EXP-248), artifact-172 (EXP-249), artifact-173 (EXP-250), artifact-174 (EXP-251), artifact-175 (EXP-254), artifact-176 (EXP-255), artifact-177 (EXP-256), artifact-178 (EXP-257), artifact-179 (EXP-258), artifact-180 (EXP-259), artifact-181 (EXP-261), artifact-182 (EXP-263), artifact-183 (EXP-264), artifact-184 (EXP-265) |
| active-reset-20260429; qwen3vl_hybrid; unvalidated topology product128; hardware unknown; GBS512 | artifact-138 (EXP-161), artifact-139 (EXP-162), artifact-140 (EXP-163) |
| active-reset-20260429; qwen3vl_hybrid; unvalidated topology product256; hardware unknown; GBS512 | artifact-141 (EXP-164), artifact-142 (EXP-165) |
| active-reset-20260429; qwen3vl; 16GB300 current-thread cohort; GBS64 | artifact-196 (EXP-PR131-MANTIS16-752807-20-GB300-legacy-window), artifact-197 (EXP-PR131-MANTIS16-752807-20-GB300) |

</details>

<details>
<summary>G. Failed or partial records — 6 artifacts</summary>

| Recorded cohort / configuration | Artifact IDs and original experiment IDs |
|---|---|
| active-reset-20260429; qwen3vl_hybrid; hardware/world unknown; GBS512 | artifact-111 (EXP-108-partial), artifact-112 (EXP-109-partial) |
| active-reset-20260429; model unknown; hardware/world unknown; GBSunknown | artifact-151 (EXP-207), artifact-152 (EXP-208), artifact-155 (EXP-212), artifact-156 (EXP-213) |

</details>


## Complete historical per-experiment inventory

All 231 historical artifacts are listed below, including failed/partial records and alternate-window artifacts. They are not 231 successful or newly corrected runs. Corrected historical TFLOPs is unavailable; original legacy rates remain in the unchanged [machine-readable audit](per-experiment.json). Five topology products of 128/256 are unvalidated and must not normalize rates. The six new accepted cells above are additional, not replacements.

<details>
<summary>Show all 231 historical artifact records</summary>

| Artifact | Namespace / experiment | Model | World / GBS / seq | Window | Status | Median step ms | Missing references | Scheduled tok/s global (per GPU) | Encoder modeled TF/GPU mean | Decoder native TF/GPU mean | Encoder + decoder modeled TF/GPU mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifact-001 | qwen3-phase4-final / EXP-040 | qwen3_30b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 4122.6 | source_pin, dataset, owned_original_log | 508696.5 (N/A) | N/A | unavailable | N/A |
| artifact-002 | qwen3-phase4-final / EXP-041 | qwen3_30b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 10899.2 | source_pin, dataset, owned_original_log | 192413.4 (N/A) | N/A | unavailable | N/A |
| artifact-003 | qwen3-phase4-final / EXP-042 | qwen3_30b_a3b | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 4602.4 | source_pin, dataset, owned_original_log | 911329.7 (N/A) | N/A | unavailable | N/A |
| artifact-004 | qwen3-phase4-final / EXP-043 | qwen3_30b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 8023.3 | source_pin, dataset, owned_original_log | 1045530.9 (N/A) | N/A | unavailable | N/A |
| artifact-005 | qwen3-phase4-final / EXP-045 | qwen3_30b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5048.1 | source_pin, dataset, owned_original_log | 415433.9 (N/A) | N/A | unavailable | N/A |
| artifact-006 | qwen35vl-phase0-3 / EXP-001 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7126.1 | source_pin, dataset, owned_original_log | 294291.7 (N/A) | N/A | unavailable | N/A |
| artifact-007 | qwen35vl-phase0-3 / EXP-002 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7038.2 | source_pin, dataset, owned_original_log | 297967.1 (N/A) | N/A | unavailable | N/A |
| artifact-008 | qwen35vl-phase0-3 / EXP-003 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7091.9 | source_pin, dataset, owned_original_log | 295710.9 (N/A) | N/A | unavailable | N/A |
| artifact-009 | qwen35vl-phase0-3 / EXP-004 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7115 | source_pin, dataset, owned_original_log | 294750.8 (N/A) | N/A | unavailable | N/A |
| artifact-010 | qwen35vl-phase0-3 / EXP-005 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7121.6 | source_pin, dataset, owned_original_log | 294477.6 (N/A) | N/A | unavailable | N/A |
| artifact-011 | qwen35vl-phase0-3 / EXP-006 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7381.6 | source_pin, dataset, owned_original_log | 284105.3 (N/A) | N/A | unavailable | N/A |
| artifact-012 | qwen35vl-phase0-3 / EXP-007 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7147.8 | source_pin, dataset, owned_original_log | 293398.2 (N/A) | N/A | unavailable | N/A |
| artifact-013 | qwen35vl-phase0-3 / EXP-008 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7173.3 | source_pin, dataset, owned_original_log | 292355.3 (N/A) | N/A | unavailable | N/A |
| artifact-014 | qwen35vl-phase0-3 / EXP-009 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7433 | source_pin, dataset, owned_original_log | 282140.7 (N/A) | N/A | unavailable | N/A |
| artifact-015 | qwen35vl-phase0-3 / EXP-010 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7233.3 | source_pin, dataset, owned_original_log | 289930.2 (N/A) | N/A | unavailable | N/A |
| artifact-016 | qwen35vl-phase0-3 / EXP-011 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7114.9 | source_pin, dataset, owned_original_log | 294755.0 (N/A) | N/A | unavailable | N/A |
| artifact-017 | qwen35vl-phase0-3 / EXP-012 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7275.5 | source_pin, dataset, owned_original_log | 288248.5 (N/A) | N/A | unavailable | N/A |
| artifact-018 | qwen35vl-phase0-3 / EXP-013 | qwen35_vl_35b_a3b | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 7147.4 | source_pin, dataset, owned_original_log | 293414.7 (N/A) | N/A | unavailable | N/A |
| artifact-019 | qwen35vl-phase0-3 / EXP-014 | qwen35_vl_35b_a3b | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8207.4 | source_pin, dataset, owned_original_log | 511039.3 (N/A) | N/A | unavailable | N/A |
| artifact-020 | qwen35vl-phase0-3 / EXP-015 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 13603.2 | source_pin, dataset, owned_original_log | 616664.3 (N/A) | N/A | unavailable | N/A |
| artifact-021 | qwen35vl-phase0-3 / EXP-017 | qwen35_vl_35b_a3b | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 9247.5 | source_pin, dataset, owned_original_log | 453560.9 (N/A) | N/A | unavailable | N/A |
| artifact-022 | qwen35vl-phase0-3 / EXP-019 | qwen35_vl_35b_a3b | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 9761.4 | source_pin, dataset, owned_original_log | 429682.6 (N/A) | N/A | unavailable | N/A |
| artifact-023 | qwen35vl-phase0-3 / EXP-027 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 13610.3 | source_pin, dataset, owned_original_log | 616342.6 (N/A) | N/A | unavailable | N/A |
| artifact-024 | qwen35vl-phase0-3 / EXP-028 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 13522.8 | source_pin, dataset, owned_original_log | 620330.7 (N/A) | N/A | unavailable | N/A |
| artifact-025 | qwen35vl-phase0-3 / EXP-029 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 14057.6 | source_pin, dataset, owned_original_log | 596731.2 (N/A) | N/A | unavailable | N/A |
| artifact-026 | qwen35vl-phase0-3 / EXP-030 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 13736.7 | source_pin, dataset, owned_original_log | 610671.3 (N/A) | N/A | unavailable | N/A |
| artifact-027 | qwen35vl-phase0-3 / EXP-031 | qwen35_vl_35b_a3b | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 13874.9 | source_pin, dataset, owned_original_log | 604588.7 (N/A) | N/A | unavailable | N/A |
| artifact-028 | active-reset-20260429 / EXP-000 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5571.7 | source_pin, dataset, owned_original_log | 376393.6 (N/A) | N/A | unavailable | N/A |
| artifact-029 | active-reset-20260429 / EXP-003 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5622.4 | source_pin, dataset, owned_original_log | 372999.4 (N/A) | N/A | unavailable | N/A |
| artifact-030 | active-reset-20260429 / EXP-004 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5641.7 | source_pin, dataset, owned_original_log | 371723.4 (N/A) | N/A | unavailable | N/A |
| artifact-031 | active-reset-20260429 / EXP-005 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5676.3 | source_pin, dataset, owned_original_log | 369457.6 (N/A) | N/A | unavailable | N/A |
| artifact-032 | active-reset-20260429 / EXP-006 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5907.8 | source_pin, dataset, owned_original_log | 354980.2 (N/A) | N/A | unavailable | N/A |
| artifact-033 | active-reset-20260429 / EXP-009 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14081 | source_pin, dataset, owned_original_log | 148934.9 (N/A) | N/A | unavailable | N/A |
| artifact-034 | active-reset-20260429 / EXP-010 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14127.4 | source_pin, dataset, owned_original_log | 148445.7 (N/A) | N/A | unavailable | N/A |
| artifact-035 | active-reset-20260429 / EXP-011 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5673.8 | source_pin, dataset, owned_original_log | 369620.4 (N/A) | N/A | unavailable | N/A |
| artifact-036 | active-reset-20260429 / EXP-012 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5701.7 | source_pin, dataset, owned_original_log | 367811.7 (N/A) | N/A | unavailable | N/A |
| artifact-037 | active-reset-20260429 / EXP-013 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5929.4 | source_pin, dataset, owned_original_log | 353687.1 (N/A) | N/A | unavailable | N/A |
| artifact-038 | active-reset-20260429 / EXP-014 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 6459.3 | source_pin, dataset, owned_original_log | 324671.7 (N/A) | N/A | unavailable | N/A |
| artifact-039 | active-reset-20260429 / EXP-015 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 5039.7 | source_pin, dataset, owned_original_log | 416126.4 (N/A) | N/A | unavailable | N/A |
| artifact-040 | active-reset-20260429 / EXP-016 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 4798.6 | source_pin, dataset, owned_original_log | 437034.1 (N/A) | N/A | unavailable | N/A |
| artifact-041 | active-reset-20260429 / EXP-017 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 8235.7 | source_pin, dataset, owned_original_log | 254641.6 (N/A) | N/A | unavailable | N/A |
| artifact-042 | active-reset-20260429 / EXP-018 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 6543.8 | source_pin, dataset, owned_original_log | 640958.5 (N/A) | N/A | unavailable | N/A |
| artifact-043 | active-reset-20260429 / EXP-019 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 12071.7 | source_pin, dataset, owned_original_log | 694898.6 (N/A) | N/A | unavailable | N/A |
| artifact-044 | active-reset-20260429 / EXP-020 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 17526.6 | source_pin, dataset, owned_original_log | 119655.4 (N/A) | N/A | unavailable | N/A |
| artifact-045 | active-reset-20260429 / EXP-022 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14041.9 | source_pin, dataset, owned_original_log | 149349.6 (N/A) | N/A | unavailable | N/A |
| artifact-046 | active-reset-20260429 / EXP-023 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 13997.9 | source_pin, dataset, owned_original_log | 149819.0 (N/A) | N/A | unavailable | N/A |
| artifact-047 | active-reset-20260429 / EXP-025 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 11922.6 | source_pin, dataset, owned_original_log | 703588.8 (N/A) | N/A | unavailable | N/A |
| artifact-048 | active-reset-20260429 / EXP-026 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 26334 | source_pin, dataset, owned_original_log | 637093.3 (N/A) | N/A | unavailable | N/A |
| artifact-049 | active-reset-20260429 / EXP-030 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 13641.7 | source_pin, dataset, owned_original_log | 153731.0 (N/A) | N/A | unavailable | N/A |
| artifact-050 | active-reset-20260429 / EXP-034 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14306.8 | source_pin, dataset, owned_original_log | 146584.3 (N/A) | N/A | unavailable | N/A |
| artifact-051 | active-reset-20260429 / EXP-035 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14285.8 | source_pin, dataset, owned_original_log | 146799.8 (N/A) | N/A | unavailable | N/A |
| artifact-052 | active-reset-20260429 / EXP-036 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 12880.5 | source_pin, dataset, owned_original_log | 162816.0 (N/A) | N/A | unavailable | N/A |
| artifact-053 | active-reset-20260429 / EXP-037 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14443.3 | source_pin, dataset, owned_original_log | 145199.0 (N/A) | N/A | unavailable | N/A |
| artifact-054 | active-reset-20260429 / EXP-038 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 12962.2 | source_pin, dataset, owned_original_log | 161789.8 (N/A) | N/A | unavailable | N/A |
| artifact-055 | active-reset-20260429 / EXP-039 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 17831.3 | source_pin, dataset, owned_original_log | 470442.9 (N/A) | N/A | unavailable | N/A |
| artifact-056 | active-reset-20260429 / EXP-040fix | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 13116.5 | source_pin, dataset, owned_original_log | 319773.1 (N/A) | N/A | unavailable | N/A |
| artifact-057 | active-reset-20260429 / EXP-041 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 14144.4 | source_pin, dataset, owned_original_log | 296534.6 (N/A) | N/A | unavailable | N/A |
| artifact-058 | active-reset-20260429 / EXP-042 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 31439.4 | source_pin, dataset, owned_original_log | 533636.6 (N/A) | N/A | unavailable | N/A |
| artifact-059 | active-reset-20260429 / EXP-043 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 32123.9 | source_pin, dataset, owned_original_log | 522265.9 (N/A) | N/A | unavailable | N/A |
| artifact-060 | active-reset-20260429 / EXP-044 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 15783.4 | source_pin, dataset, owned_original_log | 531483.0 (N/A) | N/A | unavailable | N/A |
| artifact-061 | active-reset-20260429 / EXP-045 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 31838.2 | source_pin, dataset, owned_original_log | 526952.4 (N/A) | N/A | unavailable | N/A |
| artifact-062 | active-reset-20260429 / EXP-046 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 35844.35 | source_pin, dataset, owned_original_log | 234028.7 (N/A) | N/A | unavailable | N/A |
| artifact-063 | active-reset-20260429 / EXP-047 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 42757.35 | source_pin, dataset, owned_original_log | 392382.0 (N/A) | N/A | unavailable | N/A |
| artifact-064 | active-reset-20260429 / EXP-048 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 14448 | source_pin, dataset, owned_original_log | 290303.4 (N/A) | N/A | unavailable | N/A |
| artifact-065 | active-reset-20260429 / EXP-050 | qwen3vl_hybrid | 64 / 512 / 65536 | iters 10-50 | accepted_measurement_artifact | 78602.7 | source_pin, dataset, owned_original_log | 426886.5 (N/A) | N/A | unavailable | N/A |
| artifact-066 | active-reset-20260429 / EXP-051 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 16329.4 | source_pin, dataset, owned_original_log | 256856.0 (N/A) | N/A | unavailable | N/A |
| artifact-067 | active-reset-20260429 / EXP-052 | qwen3vl_hybrid | 64 / 512 / 65536 | iters 10-50 | accepted_measurement_artifact | 78768.7 | source_pin, dataset, owned_original_log | 425986.9 (N/A) | N/A | unavailable | N/A |
| artifact-068 | active-reset-20260429 / EXP-052v4 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 14720.7 | source_pin, dataset, owned_original_log | 284925.6 (N/A) | N/A | unavailable | N/A |
| artifact-069 | active-reset-20260429 / EXP-052v5 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 15284.7 | source_pin, dataset, owned_original_log | 274411.9 (N/A) | N/A | unavailable | N/A |
| artifact-070 | active-reset-20260429 / EXP-052v6 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 19015.7 | source_pin, dataset, owned_original_log | 220570.6 (N/A) | N/A | unavailable | N/A |
| artifact-071 | active-reset-20260429 / EXP-052v7 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 23585 | source_pin, dataset, owned_original_log | 177837.8 (N/A) | N/A | unavailable | N/A |
| artifact-072 | active-reset-20260429 / EXP-053 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 17829.7 | source_pin, dataset, owned_original_log | 470485.1 (N/A) | N/A | unavailable | N/A |
| artifact-073 | active-reset-20260429 / EXP-054 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 27097.2 | source_pin, dataset, owned_original_log | 619149.4 (N/A) | N/A | unavailable | N/A |
| artifact-074 | active-reset-20260429 / EXP-054final2 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14320.5 | source_pin, dataset, owned_original_log | 146444.0 (N/A) | N/A | unavailable | N/A |
| artifact-075 | active-reset-20260429 / EXP-055fix | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14484.3 | source_pin, dataset, owned_original_log | 144787.9 (N/A) | N/A | unavailable | N/A |
| artifact-076 | active-reset-20260429 / EXP-056fix | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14559.8 | source_pin, dataset, owned_original_log | 144037.1 (N/A) | N/A | unavailable | N/A |
| artifact-077 | active-reset-20260429 / EXP-057 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 14607.4 | source_pin, dataset, owned_original_log | 287135.6 (N/A) | N/A | unavailable | N/A |
| artifact-078 | active-reset-20260429 / EXP-057fix | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 17197.1 | source_pin, dataset, owned_original_log | 243896.0 (N/A) | N/A | unavailable | N/A |
| artifact-079 | active-reset-20260429 / EXP-058 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 17447.6 | source_pin, dataset, owned_original_log | 480788.6 (N/A) | N/A | unavailable | N/A |
| artifact-080 | active-reset-20260429 / EXP-059 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 22661.6 | source_pin, dataset, owned_original_log | 740336.8 (N/A) | N/A | unavailable | N/A |
| artifact-081 | active-reset-20260429 / EXP-059fix | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14781.2 | source_pin, dataset, owned_original_log | 141879.7 (N/A) | N/A | unavailable | N/A |
| artifact-082 | active-reset-20260429 / EXP-060 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 21716.5 | source_pin, dataset, owned_original_log | 772556.2 (N/A) | N/A | unavailable | N/A |
| artifact-083 | active-reset-20260429 / EXP-060fix | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 15080.2 | source_pin, dataset, owned_original_log | 139066.6 (N/A) | N/A | unavailable | N/A |
| artifact-084 | active-reset-20260429 / EXP-061 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 25262.5 | source_pin, dataset, owned_original_log | 664115.4 (N/A) | N/A | unavailable | N/A |
| artifact-085 | active-reset-20260429 / EXP-062 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 18005.2 | source_pin, dataset, owned_original_log | 465899.2 (N/A) | N/A | unavailable | N/A |
| artifact-086 | active-reset-20260429 / EXP-064 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 25716.6 | source_pin, dataset, owned_original_log | 163097.1 (N/A) | N/A | unavailable | N/A |
| artifact-087 | active-reset-20260429 / EXP-065 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 30494.7 | source_pin, dataset, owned_original_log | 550168.3 (N/A) | N/A | unavailable | N/A |
| artifact-088 | active-reset-20260429 / EXP-070 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 7870.3 | source_pin, dataset, owned_original_log | 532928.1 (N/A) | N/A | unavailable | N/A |
| artifact-089 | active-reset-20260429 / EXP-071v3 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 12849 | source_pin, dataset, owned_original_log | 326430.4 (N/A) | N/A | unavailable | N/A |
| artifact-090 | active-reset-20260429 / EXP-072 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8492.8 | source_pin, dataset, owned_original_log | 493865.9 (N/A) | N/A | unavailable | N/A |
| artifact-091 | active-reset-20260429 / EXP-073v4 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 13120.9 | source_pin, dataset, owned_original_log | 319665.9 (N/A) | N/A | unavailable | N/A |
| artifact-092 | active-reset-20260429 / EXP-074 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 14605.8 | source_pin, dataset, owned_original_log | 287167.0 (N/A) | N/A | unavailable | N/A |
| artifact-093 | active-reset-20260429 / EXP-075v2 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 12940.2 | source_pin, dataset, owned_original_log | 324129.8 (N/A) | N/A | unavailable | N/A |
| artifact-094 | active-reset-20260429 / EXP-076v2 | qwen3 | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 7898.4 | source_pin, dataset, owned_original_log | 531032.1 (N/A) | N/A | unavailable | N/A |
| artifact-095 | active-reset-20260429 / EXP-077v8 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 18222.9 | source_pin, dataset, owned_original_log | 230166.7 (N/A) | N/A | unavailable | N/A |
| artifact-096 | active-reset-20260429 / EXP-078v7 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 22086.1 | source_pin, dataset, owned_original_log | 189907.0 (N/A) | N/A | unavailable | N/A |
| artifact-097 | active-reset-20260429 / EXP-079v4 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 23821.8 | source_pin, dataset, owned_original_log | 176070.0 (N/A) | N/A | unavailable | N/A |
| artifact-098 | active-reset-20260429 / EXP-080v2 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 21897.1 | source_pin, dataset, owned_original_log | 191546.1 (N/A) | N/A | unavailable | N/A |
| artifact-099 | active-reset-20260429 / EXP-080v3 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 25362 | source_pin, dataset, owned_original_log | 165377.5 (N/A) | N/A | unavailable | N/A |
| artifact-100 | active-reset-20260429 / EXP-083 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 30574.6 | source_pin, dataset, owned_original_log | 137182.6 (N/A) | N/A | unavailable | N/A |
| artifact-101 | active-reset-20260429 / EXP-084 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 28523.6 | source_pin, dataset, owned_original_log | 147046.8 (N/A) | N/A | unavailable | N/A |
| artifact-102 | active-reset-20260429 / EXP-085 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 26482 | source_pin, dataset, owned_original_log | 158383.2 (N/A) | N/A | unavailable | N/A |
| artifact-103 | active-reset-20260429 / EXP-090 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 25778.7 | source_pin, dataset | 162704.2 (N/A) | N/A | unavailable | N/A |
| artifact-104 | active-reset-20260429 / EXP-092 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 28645.7 | source_pin, dataset | 146420.0 (N/A) | N/A | unavailable | N/A |
| artifact-105 | active-reset-20260429 / EXP-093 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 29205.2 | source_pin, dataset | 143615.0 (N/A) | N/A | unavailable | N/A |
| artifact-106 | active-reset-20260429 / EXP-094 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 29468.1 | source_pin, dataset | 142333.7 (N/A) | N/A | unavailable | N/A |
| artifact-107 | active-reset-20260429 / EXP-095-mock | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 30386.5 | source_pin, dataset | 138031.8 (N/A) | N/A | unavailable | N/A |
| artifact-108 | active-reset-20260429 / EXP-096-mock | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 25822.6 | source_pin, dataset | 162427.6 (N/A) | N/A | unavailable | N/A |
| artifact-109 | active-reset-20260429 / EXP-099 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 51933.2 | source_pin, dataset | 80763.4 (N/A) | N/A | unavailable | N/A |
| artifact-110 | active-reset-20260429 / EXP-107 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 36753.9 | source_pin, dataset | 114118.6 (N/A) | N/A | unavailable | N/A |
| artifact-111 | active-reset-20260429 / EXP-108-partial | qwen3vl_hybrid | None / 512 / 8192 | null | recorded_partial_verifier_skipped | None | world, config, owned_original_log | N/A | N/A | unavailable | N/A |
| artifact-112 | active-reset-20260429 / EXP-109-partial | qwen3vl_hybrid | None / 512 / 8192 | null | recorded_partial_verifier_skipped | None | world, config, owned_original_log | N/A | N/A | unavailable | N/A |
| artifact-113 | active-reset-20260429 / EXP-110 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 43765.7 | source_pin | 95835.4 (N/A) | N/A | unavailable | N/A |
| artifact-114 | active-reset-20260429 / EXP-115 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 29843.7 | source_pin, dataset | 140542.4 (N/A) | N/A | unavailable | N/A |
| artifact-115 | active-reset-20260429 / EXP-116 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 42351 | source_pin, dataset | 99036.7 (N/A) | N/A | unavailable | N/A |
| artifact-116 | active-reset-20260429 / EXP-117 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 41108.8 | source_pin, dataset | 102029.3 (N/A) | N/A | unavailable | N/A |
| artifact-117 | active-reset-20260429 / EXP-118 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 34720.8 | source_pin | 120800.9 (N/A) | N/A | unavailable | N/A |
| artifact-118 | active-reset-20260429 / EXP-119 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 34952.8 | source_pin | 119999.1 (N/A) | N/A | unavailable | N/A |
| artifact-119 | active-reset-20260429 / EXP-121 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 30781.8 | source_pin, dataset | 136259.2 (N/A) | N/A | unavailable | N/A |
| artifact-120 | active-reset-20260429 / EXP-123 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 30178.5 | source_pin, dataset | 138983.2 (N/A) | N/A | unavailable | N/A |
| artifact-121 | active-reset-20260429 / EXP-124 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 30510.5 | source_pin, dataset | 137470.8 (N/A) | N/A | unavailable | N/A |
| artifact-122 | active-reset-20260429 / EXP-125 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 35031.7 | source_pin | 119728.8 (N/A) | N/A | unavailable | N/A |
| artifact-123 | active-reset-20260429 / EXP-126 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 34833 | source_pin | 120411.8 (N/A) | N/A | unavailable | N/A |
| artifact-124 | active-reset-20260429 / EXP-133 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 39248.4 | source_pin, owned_original_log | 106865.6 (N/A) | N/A | unavailable | N/A |
| artifact-125 | active-reset-20260429 / EXP-134 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 39167.4 | source_pin, owned_original_log | 107086.6 (N/A) | N/A | unavailable | N/A |
| artifact-126 | active-reset-20260429 / EXP-135 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 34612.5 | source_pin, owned_original_log | 121178.9 (N/A) | N/A | unavailable | N/A |
| artifact-127 | active-reset-20260429 / EXP-136 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8623.8 | source_pin, owned_original_log | 486363.8 (N/A) | N/A | unavailable | N/A |
| artifact-128 | active-reset-20260429 / EXP-137 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8573.6 | source_pin, owned_original_log | 489211.5 (N/A) | N/A | unavailable | N/A |
| artifact-129 | active-reset-20260429 / EXP-138 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 16604.5 | source_pin, owned_original_log | 505200.9 (N/A) | N/A | unavailable | N/A |
| artifact-130 | active-reset-20260429 / EXP-140 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8642.3 | source_pin, owned_original_log | 485322.7 (N/A) | N/A | unavailable | N/A |
| artifact-131 | active-reset-20260429 / EXP-141 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8623.6 | source_pin, owned_original_log | 486375.1 (N/A) | N/A | unavailable | N/A |
| artifact-132 | active-reset-20260429 / EXP-142 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8600.1 | source_pin, owned_original_log | 487704.1 (N/A) | N/A | unavailable | N/A |
| artifact-133 | active-reset-20260429 / EXP-143 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 9798.4 | source_pin, owned_original_log | 428060.1 (N/A) | N/A | unavailable | N/A |
| artifact-134 | active-reset-20260429 / EXP-152 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 40201.1 | source_pin, owned_original_log | 104333.1 (N/A) | N/A | unavailable | N/A |
| artifact-135 | active-reset-20260429 / EXP-153 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 39094.8 | source_pin, owned_original_log | 107285.5 (N/A) | N/A | unavailable | N/A |
| artifact-136 | active-reset-20260429 / EXP-154 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 39059.1 | source_pin, owned_original_log | 107383.5 (N/A) | N/A | unavailable | N/A |
| artifact-137 | active-reset-20260429 / EXP-160 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 43294.6 | source_pin, owned_original_log | 193756.4 (N/A) | N/A | unavailable | N/A |
| artifact-138 | active-reset-20260429 / EXP-161 | qwen3vl_hybrid | 128 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 70202.5 | source_pin, owned_original_log | 119491.6 (N/A) | N/A | unavailable | N/A |
| artifact-139 | active-reset-20260429 / EXP-162 | qwen3vl_hybrid | 128 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 69757.1 | source_pin, owned_original_log | 120254.5 (N/A) | N/A | unavailable | N/A |
| artifact-140 | active-reset-20260429 / EXP-163 | qwen3vl_hybrid | 128 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 79219.1 | source_pin, owned_original_log | 211782.5 (N/A) | N/A | unavailable | N/A |
| artifact-141 | active-reset-20260429 / EXP-164 | qwen3vl_hybrid | 256 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 117029.9 | source_pin, owned_original_log | 143358.4 (N/A) | N/A | unavailable | N/A |
| artifact-142 | active-reset-20260429 / EXP-165 | qwen3vl_hybrid | 256 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 116344.2 | source_pin, owned_original_log | 144203.3 (N/A) | N/A | unavailable | N/A |
| artifact-143 | active-reset-20260429 / EXP-172 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 16590.9 | source_pin, owned_original_log | 505615.0 (N/A) | N/A | unavailable | N/A |
| artifact-144 | active-reset-20260429 / EXP-180 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 16261.4 | source_pin, owned_original_log | 515860.1 (N/A) | N/A | unavailable | N/A |
| artifact-145 | active-reset-20260429 / EXP-182 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 16264.8 | source_pin, owned_original_log | 515752.3 (N/A) | N/A | unavailable | N/A |
| artifact-146 | active-reset-20260429 / EXP-185 | qwen3vl_hybrid | 64 / 512 / 65536 | iters 10-50 | accepted_measurement_artifact | 163754.3 | source_pin, owned_original_log | 204907.2 (N/A) | N/A | unavailable | N/A |
| artifact-147 | active-reset-20260429 / EXP-200 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8590.2 | source_pin, owned_original_log | 488266.2 (N/A) | N/A | unavailable | N/A |
| artifact-148 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 9005.8 | source_pin, owned_original_log | 465733.6 (N/A) | N/A | unavailable | N/A |
| artifact-149 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8997.7 | source_pin, owned_original_log | 466152.9 (N/A) | N/A | unavailable | N/A |
| artifact-150 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8992.3 | source_pin, owned_original_log | 466432.8 (N/A) | N/A | unavailable | N/A |
| artifact-151 | active-reset-20260429 / EXP-207 | null | None / None / None | null | recorded_failed | None | model, world, gbs, config, owned_original_log, sequence_length | N/A | N/A | unavailable | N/A |
| artifact-152 | active-reset-20260429 / EXP-208 | null | None / None / None | null | recorded_failed | None | model, world, gbs, config, owned_original_log, sequence_length | N/A | N/A | unavailable | N/A |
| artifact-153 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 9457.1 | source_pin, owned_original_log | 443508.5 (N/A) | N/A | unavailable | N/A |
| artifact-154 | active-reset-20260429 / UNKNOWN | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 9436.5 | source_pin, owned_original_log | 444476.7 (N/A) | N/A | unavailable | N/A |
| artifact-155 | active-reset-20260429 / EXP-212 | null | None / None / None | null | recorded_failed | None | model, world, gbs, config, owned_original_log, sequence_length | N/A | N/A | unavailable | N/A |
| artifact-156 | active-reset-20260429 / EXP-213 | null | None / None / None | null | recorded_failed | None | model, world, gbs, config, owned_original_log, sequence_length | N/A | N/A | unavailable | N/A |
| artifact-157 | active-reset-20260429 / EXP-230 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 8997.7 | source_pin, owned_original_log | 466152.9 (N/A) | N/A | unavailable | N/A |
| artifact-158 | active-reset-20260429 / EXP-231 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 21999.4 | source_pin, owned_original_log | 190655.4 (N/A) | N/A | unavailable | N/A |
| artifact-159 | active-reset-20260429 / EXP-232 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 39437.9 | source_pin, owned_original_log | 106352.1 (N/A) | N/A | unavailable | N/A |
| artifact-160 | active-reset-20260429 / EXP-233 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 21879.8 | source_pin, owned_original_log | 191697.5 (N/A) | N/A | unavailable | N/A |
| artifact-161 | active-reset-20260429 / EXP-234 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 15493.8 | source_pin, owned_original_log | 270708.5 (N/A) | N/A | unavailable | N/A |
| artifact-162 | active-reset-20260429 / EXP-235 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 16037.8 | source_pin, owned_original_log | 261526.1 (N/A) | N/A | unavailable | N/A |
| artifact-163 | active-reset-20260429 / EXP-236 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 39472.2 | source_pin, owned_original_log | 106259.7 (N/A) | N/A | unavailable | N/A |
| artifact-164 | active-reset-20260429 / EXP-237 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 21755.2 | source_pin, owned_original_log | 192795.5 (N/A) | N/A | unavailable | N/A |
| artifact-165 | active-reset-20260429 / EXP-238 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 16106.6 | source_pin, owned_original_log | 260409.0 (N/A) | N/A | unavailable | N/A |
| artifact-166 | active-reset-20260429 / EXP-239 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 17300.3 | source_pin, owned_original_log | 242441.1 (N/A) | N/A | unavailable | N/A |
| artifact-167 | active-reset-20260429 / EXP-240 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 26962.8 | source_pin, owned_original_log | 155558.9 (N/A) | N/A | unavailable | N/A |
| artifact-168 | active-reset-20260429 / EXP-241 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 16635.4 | source_pin, owned_original_log | 252131.2 (N/A) | N/A | unavailable | N/A |
| artifact-169 | active-reset-20260429 / EXP-246 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 21298.8 | source_pin, owned_original_log | 196926.8 (N/A) | N/A | unavailable | N/A |
| artifact-170 | active-reset-20260429 / EXP-247 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 36418.5 | source_pin, owned_original_log | 115169.6 (N/A) | N/A | unavailable | N/A |
| artifact-171 | active-reset-20260429 / EXP-248 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 20604 | source_pin, owned_original_log | 203567.5 (N/A) | N/A | unavailable | N/A |
| artifact-172 | active-reset-20260429 / EXP-249 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 35310.2 | source_pin, owned_original_log | 118784.5 (N/A) | N/A | unavailable | N/A |
| artifact-173 | active-reset-20260429 / EXP-250 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 10447.2 | source_pin, owned_original_log | 401476.4 (N/A) | N/A | unavailable | N/A |
| artifact-174 | active-reset-20260429 / EXP-251 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 10350.8 | source_pin, owned_original_log | 405215.4 (N/A) | N/A | unavailable | N/A |
| artifact-175 | active-reset-20260429 / EXP-254 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 40251.2 | source_pin, owned_original_log | 104203.2 (N/A) | N/A | unavailable | N/A |
| artifact-176 | active-reset-20260429 / EXP-255 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 26562.5 | source_pin, owned_original_log | 157903.2 (N/A) | N/A | unavailable | N/A |
| artifact-177 | active-reset-20260429 / EXP-256 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 44493.4 | source_pin, owned_original_log | 188536.0 (N/A) | N/A | unavailable | N/A |
| artifact-178 | active-reset-20260429 / EXP-257 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 29317.1 | source_pin, owned_original_log | 286133.6 (N/A) | N/A | unavailable | N/A |
| artifact-179 | active-reset-20260429 / EXP-258 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 45307.5 | source_pin, owned_original_log | 185148.3 (N/A) | N/A | unavailable | N/A |
| artifact-180 | active-reset-20260429 / EXP-259 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 28021.9 | source_pin, owned_original_log | 299359.0 (N/A) | N/A | unavailable | N/A |
| artifact-181 | active-reset-20260429 / EXP-261 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 12263.3 | source_pin, owned_original_log | 684041.7 (N/A) | N/A | unavailable | N/A |
| artifact-182 | active-reset-20260429 / EXP-263 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 33465.9 | source_pin, owned_original_log | 250661.4 (N/A) | N/A | unavailable | N/A |
| artifact-183 | active-reset-20260429 / EXP-264 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 36513 | source_pin, owned_original_log | 114871.5 (N/A) | N/A | unavailable | N/A |
| artifact-184 | active-reset-20260429 / EXP-265 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 28602.9 | source_pin, owned_original_log | 146639.1 (N/A) | N/A | unavailable | N/A |
| artifact-185 | active-reset-20260429 / EXP-PR131-ENCODER-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 4170.2 | source_pin, dataset | 251445.0 (N/A) | N/A | unavailable | N/A |
| artifact-186 | active-reset-20260429 / EXP-PR131-ENCODER-CP1-PAIR749724-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 4358.8 | source_pin, dataset | 240565.3 (N/A) | N/A | unavailable | N/A |
| artifact-187 | active-reset-20260429 / EXP-PR131-ENCODER-CP2-PAIR749724-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 4538.4 | source_pin, dataset | 231045.3 (N/A) | N/A | unavailable | N/A |
| artifact-188 | active-reset-20260429 / EXP-PR131-IMAGE1X-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 4144.3 | source_pin, dataset | 253016.4 (N/A) | N/A | unavailable | N/A |
| artifact-189 | active-reset-20260429 / EXP-PR131-IMAGE1X-CP2-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 4293.5 | source_pin, dataset | 244224.1 (N/A) | N/A | unavailable | N/A |
| artifact-190 | active-reset-20260429 / EXP-PR131-IMAGE2X-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 4151.5 | source_pin, dataset | 252577.6 (N/A) | N/A | unavailable | N/A |
| artifact-191 | active-reset-20260429 / EXP-PR131-IMAGE2X-CP2-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 4267.2 | source_pin, dataset | 245729.3 (N/A) | N/A | unavailable | N/A |
| artifact-192 | active-reset-20260429 / EXP-PR131-IMAGE4X-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 4574.4 | source_pin, dataset | 229227.0 (N/A) | N/A | unavailable | N/A |
| artifact-193 | active-reset-20260429 / EXP-PR131-IMAGE4X-CP2-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 4806.8 | source_pin, dataset | 218144.3 (N/A) | N/A | unavailable | N/A |
| artifact-194 | active-reset-20260429 / EXP-PR131-IMAGE8X-CP1-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 5571.4 | source_pin, dataset | 188206.9 (N/A) | N/A | unavailable | N/A |
| artifact-195 | active-reset-20260429 / EXP-PR131-IMAGE8X-CP2-GB300 | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 6054.7 | source_pin, dataset | 173183.8 (N/A) | N/A | unavailable | N/A |
| artifact-196 | active-reset-20260429 / EXP-PR131-MANTIS16-752807-20-GB300-legacy-window | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 7221.1 | source_pin | 145210.0 (N/A) | N/A | unavailable | N/A |
| artifact-197 | active-reset-20260429 / EXP-PR131-MANTIS16-752807-20-GB300 | qwen3vl | 16 / 64 / 16384 | [4, 20] | accepted_measurement_artifact | 7221.1 | source_pin | 145210.0 (N/A) | N/A | unavailable | N/A |
| artifact-198 | active-reset-20260429 / EXP-PR131-TFLOPS16-SEQ4096-GBS256 | qwen3vl | 16 / 256 / 4096 | iters 10-50 | accepted_measurement_artifact | 7296 | source_pin | 143719.3 (N/A) | N/A | unavailable | N/A |
| artifact-199 | active-reset-20260429 / EXP-PR7-FOURWAY-751915-pr7_baseline | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 14961 | source_pin, dataset | 70087.3 (N/A) | N/A | unavailable | N/A |
| artifact-200 | active-reset-20260429 / EXP-PR7-FOURWAY-752159-pr131_latest | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 8901.4 | source_pin, dataset | 117799.0 (N/A) | N/A | unavailable | N/A |
| artifact-201 | active-reset-20260429 / EXP-PR7-FOURWAY-752159-pr7_baseline | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 13804.5 | source_pin, dataset | 75959.0 (N/A) | N/A | unavailable | N/A |
| artifact-202 | active-reset-20260429 / EXP-PR7-FOURWAY-752159-pr7_mdp | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 11479.4 | source_pin, dataset | 91344.1 (N/A) | N/A | unavailable | N/A |
| artifact-203 | active-reset-20260429 / EXP-PR7-FOURWAY-752159-pr7_packing | qwen3vl | 16 / 64 / 16384 | iters 10-50 | accepted_measurement_artifact | 9820.5 | source_pin, dataset | 106774.2 (N/A) | N/A | unavailable | N/A |
| artifact-204 | active-reset-20260429 / EXP-bridge1 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14362.6 | source_pin, dataset, owned_original_log | 146014.8 (N/A) | N/A | unavailable | N/A |
| artifact-205 | active-reset-20260429 / EXP-bridge2 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14587.7 | source_pin, dataset, owned_original_log | 143761.7 (N/A) | N/A | unavailable | N/A |
| artifact-206 | active-reset-20260429 / EXP-va1 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 21436.9 | source_pin, dataset, owned_original_log | 293487.2 (N/A) | N/A | unavailable | N/A |
| artifact-207 | active-reset-20260429 / EXP-va2 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 36553.8 | source_pin, dataset, owned_original_log | 172115.0 (N/A) | N/A | unavailable | N/A |
| artifact-208 | active-reset-20260429 / EXP-va3 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 24283.3 | source_pin, dataset, owned_original_log | 259085.7 (N/A) | N/A | unavailable | N/A |
| artifact-209 | active-reset-20260429 / EXP-va4 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14570.9 | source_pin, dataset, owned_original_log | 143927.4 (N/A) | N/A | unavailable | N/A |
| artifact-210 | active-reset-20260429 / EXP-va5 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 25380.7 | source_pin, dataset, owned_original_log | 247883.5 (N/A) | N/A | unavailable | N/A |
| artifact-211 | active-reset-20260429 / EXP-va6 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 39772.5 | source_pin, dataset, owned_original_log | 158186.1 (N/A) | N/A | unavailable | N/A |
| artifact-212 | active-reset-20260429 / EXP-varimg2 | qwen3vl_hybrid | 64 / 512 / 4096 | iters 10-50 | accepted_measurement_artifact | 14669.1 | source_pin, dataset, owned_original_log | 142963.9 (N/A) | N/A | unavailable | N/A |
| artifact-213 | active-reset-20260429 / EXP-varlen3 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 19456.8 | source_pin, dataset, owned_original_log | 323355.1 (N/A) | N/A | unavailable | N/A |
| artifact-214 | active-reset-20260429 / EXP-varlen6 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 21527.7 | source_pin, dataset, owned_original_log | 292249.3 (N/A) | N/A | unavailable | N/A |
| artifact-215 | active-reset-20260429 / EXP-vb1 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 22704.8 | source_pin, dataset, owned_original_log | 369464.1 (N/A) | N/A | unavailable | N/A |
| artifact-216 | active-reset-20260429 / EXP-vb2 | qwen3vl_hybrid | 64 / 512 / 24576 | iters 10-50 | accepted_measurement_artifact | 27402.5 | source_pin, dataset, owned_original_log | 459188.5 (N/A) | N/A | unavailable | N/A |
| artifact-217 | active-reset-20260429 / EXP-vb3 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 33112.4 | source_pin, dataset, owned_original_log | 506674.7 (N/A) | N/A | unavailable | N/A |
| artifact-218 | active-reset-20260429 / EXP-vb4 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 36434.1 | source_pin, dataset, owned_original_log | 230240.6 (N/A) | N/A | unavailable | N/A |
| artifact-219 | active-reset-20260429 / EXP-vb5 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 46711 | source_pin, dataset, owned_original_log | 359170.6 (N/A) | N/A | unavailable | N/A |
| artifact-220 | active-reset-20260429 / EXP-vb6 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 27917.1 | source_pin, dataset, owned_original_log | 300482.8 (N/A) | N/A | unavailable | N/A |
| artifact-221 | active-reset-20260429 / EXP-vc1 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 22355.7 | source_pin, dataset, owned_original_log | 281425.1 (N/A) | N/A | unavailable | N/A |
| artifact-222 | active-reset-20260429 / EXP-vc3 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 43032 | source_pin, dataset, owned_original_log | 146204.1 (N/A) | N/A | unavailable | N/A |
| artifact-223 | active-reset-20260429 / EXP-vc4 | qwen3vl_hybrid | 64 / 512 / 8192 | iters 10-50 | accepted_measurement_artifact | 18243.3 | source_pin, dataset, owned_original_log | 229909.3 (N/A) | N/A | unavailable | N/A |
| artifact-224 | active-reset-20260429 / EXP-vc5 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 22295.7 | source_pin, dataset, owned_original_log | 282182.5 (N/A) | N/A | unavailable | N/A |
| artifact-225 | active-reset-20260429 / EXP-vc6 | qwen3vl_hybrid | 64 / 512 / 24576 | iters 10-50 | accepted_measurement_artifact | 41377.8 | source_pin, dataset, owned_original_log | 304098.1 (N/A) | N/A | unavailable | N/A |
| artifact-226 | active-reset-20260429 / EXP-vd1 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 26621.8 | source_pin, dataset, owned_original_log | 236327.2 (N/A) | N/A | unavailable | N/A |
| artifact-227 | active-reset-20260429 / EXP-vd2 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 27288.4 | source_pin, dataset, owned_original_log | 230554.2 (N/A) | N/A | unavailable | N/A |
| artifact-228 | active-reset-20260429 / EXP-vd3 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 38304 | source_pin, dataset, owned_original_log | 438001.7 (N/A) | N/A | unavailable | N/A |
| artifact-229 | active-reset-20260429 / EXP-vd4 | qwen3vl_hybrid | 64 / 512 / 32768 | iters 10-50 | accepted_measurement_artifact | 51145.1 | source_pin, dataset, owned_original_log | 328031.7 (N/A) | N/A | unavailable | N/A |
| artifact-230 | active-reset-20260429 / EXP-vd5 | qwen3vl_hybrid | 64 / 512 / 12288 | iters 10-50 | accepted_measurement_artifact | 28451.8 | source_pin, dataset, owned_original_log | 221126.8 (N/A) | N/A | unavailable | N/A |
| artifact-231 | active-reset-20260429 / EXP-vd6 | qwen3vl_hybrid | 64 / 512 / 16384 | iters 10-50 | accepted_measurement_artifact | 25960.6 | source_pin, dataset, owned_original_log | 323128.4 (N/A) | N/A | unavailable | N/A |

</details>
