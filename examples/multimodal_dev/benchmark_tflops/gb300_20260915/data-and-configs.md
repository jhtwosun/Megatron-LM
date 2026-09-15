# Data distributions and configuration evidence

Working draft: historical rates are legacy observations; corrected accounting remains unavailable until boundary proof is complete.

This report separates generator rules, prepared-corpus statistics, and consumed training geometry. None of these is interchangeable with the others. It contains no private host paths, credentials, conversation text, or dataset media. Evidence names below refer to campaign artifacts; source paths are repository-relative.

## Configuration families

The common main experiment topology is 16 GB300 GPUs, four nodes, TP1 / PP2 / decoder CP2 / EP8 / ETP1, DP4, microbatch1, global batch64, sequence length16384. This gives16 microbatches per optimizer step. Global batch counts packed containers, not raw conversations or images. Random initialization, BF16, distributed optimizer, HybridEP (32 SMs), grouped expert GEMMs, router/permutation fusion, TE cross-entropy, no FP8, no recomputation, no MTP, no virtual pipeline, and effective gradient/parameter communication overlap disabled apply to these families. Sequence parallelism is requested but disabled at TP1. Encoder CP and decoder CP are different axes.

The measured native PR7/PR131 model is a Qwen3 decoder plus Qwen3.5-VL vision hybrid/proxy, not a claim of official canonical Qwen3-VL architecture. Decoder:48 layers, hidden2048,32 query heads,4 KV groups, KV channel128,128 experts/top8, expert FFN768, no shared expert. Vision:27 layers, hidden1152, FFN4304, patch16, temporal patch2, spatial merge2. Patch input width1536. Trainer vocab argument248320 becomes248321 including EOD and is padded to248448 in the resolved model; an early printed padded_vocab_size=None is not the resolved embedding size. Native real-data image-pad ID248056 is distinct from this padding calculation.

| Family | Data and encoder behavior | Graphs / reusable bridge scratch | Measurement protocol |
|---|---|---|---|
| Initial PR131 encoder-CP pair | Reference lognormal scenarios; fixed encoder CP1 versus2; fused retain, patch cap131072 | Graph scopes attention/router/preprocess, warmup2; scratch enabled both |20 steps; primary4–20, supplemental10–20 |
| Image-area sweep | Fixed one-image/4096-token raw records; four sizes; encoder CP1 versus2 at each size; fused retain/cap131072 | Same graph/scratch stack as initial pair |20 steps per cell; same windows |
| Native four-way mock, job752159 | PR7 ordinary, PR7 nonfused MDP, PR7 fused MDP, PR131 fused MDP; encoder CP1 throughout | No graphs; scratch override unset |50 unprofiled steps,10–50; separate20-step diagnostic profiles |
| Mantis, job752807 | Energon real conversations; PR131 fused MDP retain/cap131072; encoder CP1 | No graphs; scratch override unset |20 training-only steps, LR warmup2/decay20, eval0; primary4–20, supplemental10–20 |

PR7 source base: `e1484af4f5e9e5723105f731fb555dba9a32fecb`; PR131 base: `57a5c2239242340cad5c22a8dd3fec18b16015e9`. CP experiments have additional isolated CP/buffer changes; the native four-way and Mantis sources instead share only the missing NVTX import and ordinary grid-normalization compatibility patch. Do not silently treat these source stacks as identical. Runtime environment was the qualified PyTorch26.07 container plus the campaign overlay; environment and normalization toggles must follow each sealed launcher, not be inferred from model arguments alone.

The four-way variants differ in MDP mode and fused packing deliberately. The ordinary/nonfused cap is0; the fused-window cap131072 must be interpreted using the actual window builder, not as a global image count. The reviewed fixed sweep reports cap evaluation before owner filtering and1/1/2/4 fused packs as image area increases. All four unprofiled measurements completed; only ordinary and nonfused MDP profiles completed. The outer job timed out during packing-profile startup; latest profile never started. No complete four-way profile result exists.

## Native four-way mock: source rules, not an observed histogram

`examples/multimodal_dev/data/mock_mdp.py` uses a per-index Python RNG with training seed1729+index (validation/test offsets100000/200000). Each raw record draws an integer image count uniformly from1–6. Each image independently draws one integer side from224–512 inclusive and is square before normalization. Runtime image-size/pixel caps are all0; `BlendDataset._resize_hw` floors each side to a multiple of32. Thus normalized sides are224,256,...,512; these bins are not uniform: the512 bin contains only the single original draw512, while preceding bins each contain32 integer draws. Images materialize deterministic FP32 uniform[-1,1) patch tensors, not real photos.

The text recipe draws192–768 random integer IDs plus two leading IDs, hence194–770 text IDs before image-marker insertion and packing. The configured NullMultimodalTokenizer actually converts whitespace-separated integers directly (`megatron/core/tokenizers/vision/libraries/null_multimodal_tokenizer.py`), so this is not natural-language subword tokenization. Image slots and markers add to decoder content length; the text-ID range is not the total raw multimodal document length.

`pack_samples_per_item=4` with scan multiplier1 sets a raw-record start stride of4. `BlendDataset._build_packed_item` scans consecutive records until the next padded document exceeds the16384 budget; it does not stop after four documents. Static THD packing is disabled in the native four-way run. Adjacent packed items can therefore reuse overlapping raw index ranges. The deterministic recipe allows offline reconstruction if the exact dataset lengths, sampled indices, wrapping, padding, and revision are reproduced. Such reconstruction is not a captured consumed-record manifest.

Job752159 does not have the required per-iteration ref_geom distribution logs. Exact consumed raw-document counts, image counts/resolution histograms, T/U/R/A, and final per-segment padding distributions are unavailable from its retained scalar training logs. Do not describe it as fixed448, one image, or exactly four raw documents per pack. Matching generator settings alone is insufficient proof of identical consumed geometry across variants.

## Reference CP data and fixed image sweep

The earlier CP pair overrides native mock generation with a64-scenario deterministic reference pool (seed2026): configured lognormal total length, min512/max4096, target mean2048, sigma1.1. Every fifth scenario is text-only; multimodal scenarios have1–3 items with variable even patch-grid dimensions and a source rule giving approximately25% multi-frame items before budget-dependent adjustment. These are generator rules, not measured corpus proportions. Reference token IDs are generated directly; image tensors are deterministic constant sentinels, not uniform random pixels. Reference scenarios therefore must not be conflated with the native four-way generator.

The fixed sweep uses a separate explicit fixed-grid scenario: one still image, total4096 decoder tokens per raw record, one image-start marker, and remaining text split before/after the image. The qualified fixture is intended to form four such records per16384-token packed bin. The content geometry is checked; final logical and physical attention boundaries still need separate replay proof. Text decreases as image area increases; decoder total tokens are held fixed, not text content.

| Area multiplier | Pixel H×W | Grid T,H,W | Raw patches/image | Merged image tokens | Text tokens/raw record |
|---:|---|---|---:|---:|---:|
|1|448×448|1,28,28|784|196|3899|
|2|448×896|1,28,56|1568|392|3703|
|4|896×896|1,56,56|3136|784|3311|
|8|896×1792|1,56,112|6272|1568|2527|

The consumed geometry checker tests every logged iteration against global T=1,048,576 and U=4,294,967,296, R=200,704×multiplier, A=157,351,936×multiplier². This is stronger evidence than an unexecuted matrix specification. CP pairs require equal observed tuples within each size. Rectangle rows vary area exactly; they are not a square-resolution-only sweep or a long-video experiment. Aggregate geometry still does not archive every consumed image identity or prove numerical training-quality equivalence.

## Mantis prepared corpus: empirical statistics

Source: Mantis-Instruct `llava_665k_multi`, first256 selected rows, revision `01a9edfe0bb8c2582431308c5b2645a9f4796939`. Conversion retained236 training and20 validation rows with0 skipped. The split is prepared locally, not an assertion about the upstream full-corpus distribution. The pinned local Qwen3.5 tokenizer assets use revision `59d61f3ce65a6d9863b86d2e96597125219dc754`; the Energon native cooker uses these real tokenizer/template assets even though the trainer interface prints NullMultimodalTokenizer.

All256 rows passed native metadata/token preencoding at16384 against a roomy65536 budget, with matching IDs, shifted labels, masks, and geometry and no overlength/failure. This preflight did not materialize every pixel or run a model. Token counts below include actual native template/image slots; they are neither lexical word counts nor final padded THD lengths.

| Statistic | Training | Validation |
|---|---:|---:|
|Raw conversations|236|20|
|Native tokens min / mean / median / p90 / max|361 /1280.559 /838 /2520 /3221|394 /1431.900 /1211.5 /2502.3 /2618|
|Conversations with1/2/3/4 images|126 /31 /37 /42|8 /3 /6 /3|
|Image occurrences|467|44|
|Resized pixels/image min / mean / median / max|202752 /278635.443 /266240 /409600|226304 /276154.182 /266240 /368640|

Percentiles use linear interpolation at0.9×(N−1). Image statistics count occurrences, not unique visual content. Explicit image bounds were200704/1003520 pixels, patch16/merge2/temporal2; dimensions are32-aligned and aspect ratios vary. The bounds are processor inputs, not a universal exact output-area contract. Original bytes are preserved by preparation; resizing intentionally changes model inputs.

Energon uses worker0, shuffle buffer16, packing buffer16, prefetch1, max_samples_per_sequence4, physical microbatch1. This last value is not a four-document cap: bounded CPU pack checks observed training packs with11 and5 documents,24 and11 images; validation packs11 and5 documents,23 and15 images. Those four packs are samples of loader behavior, not the complete formal consumed distribution. Small-corpus training repeats;20×64=1280 optimizer-visible packs is not1280 distinct conversations.

## Mantis consumed training geometry, iterations4–20

Job752807 completed20 finite ordered steps with zero skipped/NaN iterations and eval0. The primary window contains17 steps. Logged ref_geom fields are per-bin means; multiplying byGBS64 gives the following global content geometry. Decimal log precision means these are reconstructed from rounded measurements, not an exact integer archive.

| Field | Meaning | Min | Mean | Max |
|---|---|---:|---:|---:|
|T|Sum of unpadded document content lengths|642890|671573.176|709233|
|U|Sum of squared document content lengths|1123787478|1190615801.294|1270087963.000|
|R|Sum of raw vision patches|1082996|1129626.353|1190752|
|A|Vision per-frame squared spatial patch-length sum|1203212432|1255037406.118|1325719296|

Mean content T is10493.331 per pack, not the configured padded capacity16384. Global padded token capacity is1,048,576 per optimizer step. Content U cannot recover the final aligned/padded THD attention segments. R/A likewise do not identify image-count or image-resolution histograms uniquely. Exact consumed source IDs, documents per pack, image counts/resolutions, and padding by segment were not archived for every step. Prepared-corpus statistics above must not be relabelled as the consumed weighted histogram. Whole-step time includes vision, decoder, communication, and optimizer; a vision FLOP numerator divided by that time is not encoder-only throughput.

## Nemotron preparation only

Nemotron Image v3 Turing revision `7656391d4d4cb11ec3722b34f10d499435de0460`:193 prepared rows,181 train/12 validation, one image per row. Native all-row token preflight passed without overlength at16384. Training token min/mean/median/max=1112/6672.740/6335/16178; validation=2340/7994.667/6683/15160. Training resized image area min/mean/median/max=193536/689666.829/698368/1007616 pixels. Rounding can place realized areas slightly outside nominal bounds. Two training and two validation CPU packs passed, with document counts3/1/2/1. No Nemotron full-model training, consumed runtime distribution, or throughput result exists in this evidence set.

Turing preparation verified payload SHA/size and retained its CC-BY-4.0 README. Mantis preparation validated selected payloads, not the full upstream image archive; retain subset and underlying-source license provenance rather than assuming a universal license for every image. This public report contains no dataset media or raw conversations.

## Evidence fingerprints

SHA-256 values identify evidence without exposing private locations:

- Mantis native preflight: `6a03de2de753d367e25be12e0bd12d31b1775ca5299a0d0a3d98524c8fac0499`.
- Nemotron native preflight v2: `d792ab76cab87f798474b873fdbf913983a6c0f9566cc2a1679c31d7271cd114`.
- Mantis preparation receipt: `fdfa1e70b139be8f53d4bbff421600114fcd2e0a7c0f8c13e2d5b642bbafccbe`.
- Nemotron preparation receipt: `6d131071284acd25b30e3b5a093edf3fd4db0b66d0fdc95e57f44fab628849e5`.
- Mantis primary geometry/metrics artifact: `526617bd555e0e84911e87fbacbd285c45817dd046b2b636419394216ca2435d`.

Evidence collection here was read-only over owned source/configuration/receipts and existing logs. No corpus fetch, scheduler operation, source modification, or external publication was performed.
