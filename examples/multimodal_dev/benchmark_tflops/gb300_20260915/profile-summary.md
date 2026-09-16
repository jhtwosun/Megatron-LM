# Four-way profile evidence: completed continuation

Diagnostic evidence only. These traces do not replace the accepted unprofiled measurements or establish corrected TFLOPs, hardware utilization, a pacing rank, or a performance winner.

## Completion and configuration

PR7 fused packing job **753568** and PR131 fused latest job **753569** completed with exit `0:0`. Each has 20 ordered finite training records, zero skipped/NaN iterations, no fatal log exception, and four finalized SQLite files. All eight files passed read-only integrity and stability checks; each contains four GPU workers, giving 16 workers per cell. Exact measured-source seals and separate harness-provenance checks passed.

Both use 16 GB300 GPUs, TP1/PP2/decoder CP2/EP8/ETP1, MBS1/GBS64, sequence length 16384, BF16 HybridEP, native variable mock inputs, MDP fused-retain, a 131072 encoder-window cap, and no CUDA graphs or recomputation. These are the existing PR7/PR131 hybrid/proxy launchers, not an assertion of official model-architecture identity.

The new profiles use 20 iterations, LR warmup/decay 2/20, and evaluation disabled. Older baseline/nonfused profiles from job 752159 use LR 5/50 and default evaluation. The new jobs also ran on different racks. Capture is configured as 5–8; interpreting that as displayed iterations 6–8 is source-inferred, because explicit iteration/ProfilerStep NVTX anchors were absent. There is one profile per cell, without a variance estimate.

## Observations, not a speed ranking

All sums below aggregate four worker processes per node. They overlap and must not be divided by a capture span to claim wall-time fractions or utilization.

| Cell | Worker kernel span (s) | Selected vision outer ranges per node | Forward-bridge kernel sums, nodes 0–3 (s) | Dispatch-preprocess CPU synchronization sums, nodes 0–3 (s) |
|---|---:|---|---|---|
| PR7 MDP off, 752159 | 64.43–64.60 | Outer range absent; vision still executes | Not applicable | 13.121 / 11.451 / 21.696 / 10.880 |
| PR7 nonfused MDP, 752159 | 66.83–66.98 | 192 each | 2.870 / 20.881 / 9.467 / 5.976 | 16.011 / 11.509 / 23.513 / 9.314 |
| PR7 fused packing, 753568 | 40.10–40.23 | 48 each | 0.319 / 0.466 / 1.785 / 0.715 | 3.768 / 8.482 / 2.748 / 2.099 |
| PR131 fused latest, 753569 | 47.98–48.10 | 48 each | 0.738 / 0.885 / 2.476 / 4.096 | 29.452 / 15.089 / 3.643 / 3.394 |

The 192-to-48 range change reflects invocation/fusion granularity, not proof of four times fewer images, patches, or attention operations. Unequal capture spans and LR/evaluation/rack differences make percentage speedups from this table invalid.

### Five diagnostic questions

1. **Dominant observed costs:** HybridEP device synchronization and NCCL communication lead GPU event-duration sums. Packing shows device-sync sums of 13.97–22.86 seconds per node and SendRecv sums of 9.35–13.01 seconds. Latest shows device-sync sums of 38.84/22.90 seconds on nodes 0/1 and SendRecv sums of 41.12/40.02 seconds on nodes 2/3. These are overlapping aggregate event times, not encoder-only or whole-step costs.
2. **Steady or tail-heavy:** Both occur. Latest node-0 dispatch-preprocess stream waits have representative process medians of about 4.65–4.69 ms but maxima of 502–533 ms. Nodes 2/3 also show recurring device waits with representative medians near 153–156 ms and maxima up to 1045 ms. Packing has smaller medians but substantial tails too.
3. **Phase ownership:** Kernel launches match CPU APIs by process identity and correlation ID, then use the smallest enclosing selected NVTX range on the launching thread. All 4,220,973 packing kernels and 4,277,998 latest kernels matched. Latest node-0 AllGather totals 30.28 seconds, but only 0.738 seconds maps to the named forward bridge; assigning all AllGather work to that bridge would be wrong. Backward bridge ownership is not established by an explicit backward range. Interpolation-associated GPU sums remain about 45–46 ms per node; larger enclosing CPU spans do not make interpolation GPU compute the dominant cost.
4. **Pacing rank or stage:** Unproven. Large device waits often lie outside the selected phase vocabulary. A waiting rank need not have caused the delay. No cross-node timestamp critical path or hardware-utilization claim is supported.
5. **Next evidence:** Establish the source owner and dependency of the out-of-vocabulary device-sync tails before choosing a code change. Validate any performance hypothesis with matched unprofiled repetitions; preserve unknown ownership and accounting until then.

The actual Nsight 2026 synchronization enumeration is: 1 event synchronize, 2 stream-wait-event, 3 stream synchronize, 4 context synchronize. Analysis uses that schema, not older generic numbering. Raw event names were preserved; no fixed-four-pack step grouping or cross-node clock alignment was assumed.

## Retained evidence fingerprints

Raw traces and detailed local artifacts are not bundled. These SHA-256 values bind the retained evidence used for the reviewed summary, without publishing private paths or large traces.

| Artifact | SHA-256 |
|---|---|
| Final completion/integrity receipt | `a3e9d1ef4f81dfb4cd5b84a991d2cd0eb9c1c6dfed5dffe9d822c754d54ccc47` |
| Packing SQLite summaries | `498ff0ac5f3c3b8845ee30bee1add9b2aab195a972f1ca61e2dd450434928990` |
| Packing process-aware correlation | `d74824ae7d1596bdbe3f94d4a780bafdc0146f7ddc48687b9d9a1c075ff57aa8` |
| Latest SQLite summaries | `1be3419731e62096b3f48683da62bb555cc6816e1b4f2ab3e89b1b6a19c935a2` |
| Latest process-aware correlation | `f7a0b768f1aefe6010402b0d052bce3728d76d9965587b833ed9cd56ccf67926` |

Independent parent review rechecked kernel-match counts and phase sums. The secondary-model gateway remained unavailable, so that review is a disclosed same-model fallback, not cross-model verification. Corrected accounting remains unavailable; the Draft PR is not complete merely because these profiles finished.
