# Accounting utilities: tested arithmetic, unexecuted historical replay

The three adjacent Python utilities are deliberately narrower than an accounting verdict. Seven standard-library tests pass, and four identical-input evaluations of the pinned PR7/PR131 native FLOPs functions agree exactly. The native dataset replay has **not** completed an allocated end-to-end run. Historical input identity, executed attention-boundary proof, and corrected historical rates remain unavailable.

## Scope and trust

- `native_accounting.py` extracts the exact native FLOPs function and two required helpers from separately SHA-256-checked source files. It does not import the training stack. This executes trusted Python AST, not sandboxed untrusted code: independently review and pin every input source first.
- `replay_fixed_boundaries.py` uses the original native fixed-reference dataset, single sampler and prepacked-batch normalizer. It requires an allocation and the matching native environment; it constructs no model and is not a Transformer Engine execution test. Its allocation-variable check is an operator guard, not a security boundary.
- `test_native_accounting.py` exercises arithmetic, missing evidence, hash checks, world decomposition, malformed seals, sampler consistency and rejected resumed/budget-mismatched manifests. It imports neither Torch nor a model.

The replay supports the reviewed workers-0, MBS-1, single-sampler, zero-consumed, static fixed-grid path only. Other datasets, nonstatic boundary semantics, resumed runs and different native APIs need a separate audit. A historical encoder-CP extension is not interchangeable with a different upstream constructor merely because both use encoder CP1.

## Standard-library proof

From this repository root:

```bash
python3 examples/multimodal_dev/benchmark_tflops/test_native_accounting.py
```

The separately checked native function's extracted-source SHA-256 is `d900fc09ea21b6aa42e3879cb47f729650c64e6076116faf86dd34b7e3dabd8e`. Full PR7 and PR131 training files differ; function equality does not prove model, argument, input or timing equality.

## Inputs required before accounting

Obtain the original measured source, full effective resolved arguments, ordered original timing rows and independently established global attended moments. Do not borrow geometry or a formula from another experiment. The resolved world must agree with TP × PP × CP × DP and the explicit normalization world. Missing moments return unavailable rather than an estimated rate.

`native_accounting.py --help` lists the required paths and expected hashes for the training function, utility helper and experimental-attention helper. Its rows contain `iteration`, `step_ms`, `Tpad` and `Upad`; they must cover the ordered original iteration sequence. Specify the actual inclusive measurement window, not a stale verifier label. The tool reports mean stepwise TFLOPs and pooled work/time separately. Both are modeled decoder rates, not hardware counters or complete vision FLOPs.

Equal final logical/storage cumulative arrays yield unambiguous segment moments for this adapter: nonzero dummy intervals count, repeated endpoints contribute zero. Unequal arrays remain unavailable pending backend-valid-length interpretation. Integer bounds `T <= U <= T²` reject impossible pairs but do not replace evidence that those are the executed attention lengths.

## Candidate native replay, not a completed recipe

Prepare and independently review a manifest containing full `resolved_args` and exact `dataset_kwargs`, plus `source_seal`, its expected `source_seal_sha256`, `reference_config`, `world_size`, `dp_size`, `gbs`, `total_steps`, `sampler_total_samples`, and `consumed_samples`. No default model or dataset values are supplied here. The nonempty, unique source inventory must contain the actual modules imported or extracted by this replay. Its expected digest must come from an independently reviewed inventory, not be invented after selecting arbitrary files.

The tool checks every sealed file before and after replay. It checks world, sampler horizon, original GBS/steps, dataset capacity/sequence budget and exact emitted sample indices. Preserve JSON numeric types when generating canonical argument hashes; serializing `1.0` as `1` can change a Python canonical hash despite numerical equality.

Inside an explicitly approved allocation with the matching native dependencies and read-only source/manifest mounts:

```bash
python3 examples/multimodal_dev/benchmark_tflops/replay_fixed_boundaries.py \
  --source <original-sealed-source-root> \
  --manifest <reviewed-replay-manifest.json>
```

These are placeholders, not a bundled runnable manifest. No private path, data, credential, checkpoint or cluster allocation command is provided. Keep source bytecode disabled or use read-only mounts. Capture the emitted JSON as candidate evidence, then independently bind it to the original log, job, source and consumed-input sequence. Output intentionally leaves `historical_input_identity` and `executed_TE_boundary_proof` null. Do not publish corrected historical rates until those separate gates are resolved or explicitly justified limitations are accepted.
