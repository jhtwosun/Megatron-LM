# Native Megatron accounting and historical corrections

This supplements the original reproduction snapshot; it does not rewrite its
raw measurements. The native decoder estimate, useful-content decoder estimate,
and separate vision estimate are different quantities. None is a hardware FLOP
counter, MFU, or isolated decoder timing measurement.

## Pin the executed implementation, not a model name

The audited PR7-descendant source starts at PR131 commit
`57a5c2239242340cad5c22a8dd3fec18b16015e9`, with separately recorded runtime fixes.
The audited `megatron/training/training.py` SHA256 is
`04f64365a6c132a628536c12bc3f2c2b47508c177fb5bc495f75aa15b74b3c8d`.
This is a pinned historical implementation, not a claim about latest upstream.
Archive the complete source diff/file hashes and resolved runtime arguments.
Extract `num_floating_point_operations` from that file with Python AST; execute
only trusted, hash-checked source. Do not import the training stack on a login
node. The campaign's `audit_pr7_native_flops_ast.py` checks 34 same-input rows
against the content estimator (maximum absolute difference below 1e-10 TF/GPU).
That proves formula agreement on those inputs, **not** padded-input agreement.

Resolve vocabulary after tokenizer padding and model configuration hooks. This
benchmark is a PR7 Qwen3/Qwen3.5-VL hybrid, not an official canonical Qwen3-VL:
48 standard-attention decoder layers, H2048, 32 query heads of dimension128,
4 KV groups, 128 experts/top8/FFN768, SwiGLU, no shared expert/MTP, padded
vocabulary248448. Vision is 27 layers/H1152/FFN4304 with one spatial merger,
no DeepStack. Audit GQA, KV channels, MoE layer pattern, shared/latent experts,
MTP, attention variants and all other native formula arguments for every run.
Do not infer these from a recipe comment or substitute another model's defaults.

## Count the final THD segments

For each logical packed bin, read the final cumulative sequence boundaries
actually supplied to attention. Let `l_j = cu[j+1] - cu[j]`. Count every real
document's padded length and the separate nonzero dummy tail. Repeated endpoints
are zero-length slots and contribute zero. In this audited CP2 path documents
are aligned to64 tokens; this is a configuration fact, not a universal constant.

Over every logical DP stream and microbatch in optimizer step i, compute
`Tpad_i = sum(l_j)` and `Upad_i = sum(l_j*l_j)`. Do not square the whole bin,
extend the last real document into the dummy, or substitute content T/U.
Do not sum duplicate CP/TP/PP observations as additional samples. Here
`GBS = MBS * dense_DP * num_microbatches`; W16/PP2/CP2 uses DP4 and16
microbatches, while W32/PP2/CP2 uses DP8 and8 at GBS64/MBS1.

Pass `(resolved_args, Tpad_i, Upad_i)` to the pinned native function. Its
standard-transformer branch uses the native causal half-square convention;
retain the exact coefficient and training multiplier from source. Do not
replace it with a generic `6*N*tokens` rule or an unqualified `L*(L+1)/2`
attention count. Native model accounting does not measure implementation-specific
recompute, padding kernels, communication, optimizer, or all nonmatmul work.

Content geometry remains separately useful: T=sum(original retained lengths),
U=sum(original length squared), R=sum(t*h*w), A=sum(t*(h*w)^2).
T includes visual and unsupervised tokens, not just loss-bearing tokens.
The existing `workload_flops.py` vision formula remains a separate useful-content
estimate, excluding internal head-dimension padding and extra recomputation.
If adding native padded decoder and content vision, label the mixed modeled
total explicitly; never label it hardware utilization.

## Preserve the denominator and statistic

Use each original **whole multimodal optimizer-step** wall time `s_i` in seconds
and original world size W, including in-step data waits, encoder, decoder,
communication and optimizer work. Do not use encoder-only time, median time as
every row's denominator, replay time, or microbatch time.

* Stepwise decoder rate: `F_i / (W * s_i * 1e12)`.
* Mean TF/GPU: arithmetic mean of those stepwise rates.
* Pooled TF/GPU: `sum(F_i) / (W * sum(s_i) * 1e12)`.
* Pooled content tokens/s: `sum(Tcontent_i) / sum(s_i)`.
* Pooled padded tokens/s: `sum(Tpad_i) / sum(s_i)`.

Mean per-step token rate is another statistic; label it separately. Divide global
token throughput by W only when reporting per-GPU throughput. Keep mean and
median step times distinct. Current20-step results use4–20/17 rows and optional
10–20/11; original50-step results retain10–50/41. Supplementary4–20 extraction
does not change an old run's protocol. Account for the native logger's special
iteration2 average of iterations1+2 when matching geometry; do not apply that
averaging to ordinary primary-window rows.

## Correct old results without retraining

1. Preserve raw log bytes/hash, scheduler/model/collector status, original logged
   TF, original per-step times, world and window. Write a new supplemental JSON.
2. Recover original source, final args, data manifest and bytes, tokenizer/chat
   template, seed and consumed-sample state, DP layout, workers, prefetch,
   shuffle/packing buffers, capacities, packing algorithm and library versions.
3. Replay the exact native loader without model execution for **all logical DP
   streams**, preserving ordering and worker behavior. Run Torch/real data work
   only in an allocation. Pixel-decode omission is valid only after showing it
   cannot alter packing, skips, RNG or sample order for that source.
4. Save ordered hashes of tokens, labels, masks, positions, grids, descriptors
   and cumulative boundaries, plus T/U/R/A and padded T/U per stream/step.
   Compare original ordered hashes where available. Match all original logged
   T/U/R/A rows within their documented print precision before accounting.
   Aggregate geometry alone is not token/pixel identity: historical runs without
   original ordered hashes retain that limitation even if replay hashes exist.
5. Evaluate the pinned native function with replayed padded T/U and original
   times. Record source/data/script/raw hashes, evidence scope and both rate
   statistics. A mismatch or unavailable exact replay means **unavailable**,
   not a guessed correction factor. Never transfer geometry between datasets,
   sequence lengths, workers, buffers, seeds or settings.

## Campaign reproduction commands and evidence

These commands refer to the separately preserved research workspace, **not files
bundled in this PR**. Set CAMPAIGN to that workspace; inspect each pinned script
and its required original paths/environment before execution. No new training
is needed. Native real-loader replay must use its original qualified container,
Energon7.3.2, source/overlay, data aliases and allocated `srun --overlap` wrapper.

```bash
export CAMPAIGN=/path/to/autoresearch-megatron
python3 "$CAMPAIGN/shared-state/reports/audit_pr7_native_flops_ast.py"
# Metadata-only mock replay requires the recorded NumPy2.4.4 and source artifacts.
python3 "$CAMPAIGN/shared-state/reports/replay_pr7_mock_packing.py" > mock-replay.new.json
# Inside the original qualified allocated environment (not a login node):
python "$CAMPAIGN/shared-state/reports/replay_pr7_real_packing.py" --preflight
python "$CAMPAIGN/shared-state/reports/replay_pr7_joint_loader_packing.py" --preflight
python "$CAMPAIGN/shared-state/reports/replay_pr7_joint_loader_w32_packing.py" --preflight
# After preflight/provenance review, inside that same allocation/environment:
python "$CAMPAIGN/shared-state/reports/replay_pr7_real_packing.py" > real6-replay.new.txt
python "$CAMPAIGN/shared-state/reports/replay_pr7_joint_loader_packing.py" > joint-w16-replay.new.txt
python "$CAMPAIGN/shared-state/reports/replay_pr7_joint_loader_w32_packing.py" > joint-w32-replay.new.txt
```

For portable new measurements, the existing [run/collection commands](README.md#commands)
and `collect_workload.py` remain useful-content reporting only; they do not
automatically reconstruct padded decoder geometry. Do not relabel their output.

### Historical correction ledger (2026-09-15)

All rates below are decoder TF/GPU means over original4–20; vision stays separate.
This is a provenance/accounting ledger, not a causal performance comparison.

| Family / original jobs | Exact replay settings | Native padded D mean | Evidence |
|---|---|---|---|
| Mock16K:418678,418724,418727,418731,418858,418859,418909,418910 | Native64-scenario/seeded packing, workers0, W16, exact PP receipts | 202.321,186.086,308.613,312.293,269.896,291.519,318.381,314.244 | REPLAY-PR7-MOCK-S16K-PACKED-GEOMETRY.json; all20 anchor and17 rows/job T/U/R/A; metadata replay only |
| Original real:418743,418747,418866,418891,418816,419015 | Blend3, workers1/prefetch1/packing128/shuffle128; separate8K/16K DP streams | 34.637,100.153,73.148,79.592,93.040,109.966 | REPLAY-PR7-REAL6-NATIVE-PADDING-20260915.json; all20 geometry rows/job |
| Joint loader:419358,419668,419669,419671 | Blend3 W16/8K, workers4/prefetch2/packing512/shuffle512 | 55.712,101.702,72.494,82.581 | REPLAY-PR7-JOINT-LOADER4-NATIVE-PADDING-20260915.json |
| Joint loader:419692,419707 | Same joint settings, separate W32/16K/DP8 replay | 92.345614,114.131975 | REPLAY-PR7-JOINT-W32-NATIVE-PADDING-20260915.json |
| Mantis-only420497 | Different dataset; exact padded replay not completed | **Unavailable** | Content D88.357926 + V35.584742 only; never substitute blended geometry |

Real6 raw replay SHA256:
`c7414bf40736a3888837fc90313c3edaf74652f4427d927e867afaaa642747b2`.
Joint W16 raw replay SHA256:
`22f93c2d6062917c28187dda30cd8b8171d553c213a0ce510b45642056129042`.
Joint W32 raw replay SHA256:
`7ebb25ed934de96ed446e6a82cd25c13c648f44dc7de86476dabcdc042f72b1a`.
Campaign reports `PR7-W32-PADDING-BODY-REVISED-20260915.md` and
`PR7-MANTIS-ONLY-420497.md` retain the complete timing/content/pooled tables.
Artifacts are campaign-local, not asserted to be downloadable from this PR.

419015 retains training-completed/failed-batch/recovered-collector status.
The six joint-loader jobs completed successfully but required supplemental
external-collector recovery; Mantis420497 likewise preserves its collector
failure and exact-byte recovery. Never silently repair raw UTF8, infer success
from partial rows, or turn an OOM boundary into a performance result.
