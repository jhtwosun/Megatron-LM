# Historical diagnostic evidence — 422790

This is a diagnostic result, not a throughput comparison or new checkout
qualification. The unchanged real-data eight-step run completed successfully;
all eight T/U/R/A workload rows matched the preceding real run, losses/gradients
were finite with skipped/NaN counts zero, and GraphLaunch count stayed2304/rank.
Sixteen rank receipts recorded native EP membership and clock brackets.

Under an explicit conditional relative-drift envelope of at most1000ppm,
the first-microbatch/layer1 late CP pairs were0/1 (step5),4/5 (steps6/7).
They belong to CP×PP families0,1,8,9 and4,5,12,13, respectively.
Pair members are not individually distinguishable as unique stragglers.

| Step | Late representative | Last linked original-vision GPU end | Last linked embedding GPU end | Routing GPU entry |
|---|---:|---:|---:|---:|
|5|0|3516.30ms|3529.59ms|3533.71ms|
|6|4|3149.49ms|3227.70ms|3231.95ms|
|7|4|2908.18ms|2909.03ms|2919.62ms|

Times are calibrated midpoints relative to rank0 step entry, not pure compute
durations. Last participants' matched routing AllGather lasted about0.03ms;
early peers lasted up to1517.38/1166.54/849.81ms. Step5 rank6 reached routing
at2016.35ms after embedding ended2005.48ms, then finished near3533.73ms,
when the late family entered. This supports upstream original-vision/window
delivery arrival imbalance, not seconds of AllGather payload transfer.

Exact consumed window IDs establish that steps6/7 CPU materialization finished
roughly10s before the step. Step5 materialization was outside capture. The
fused sidecar prepares the whole16-item window before pipeline forward; these
are window-wide endpoints, not first-sample-only image costs.

The path is narrowed, not fully explained: rank0 step5 pack2 had866.144526ms
from CPU range entry to its first positively linked GPU kernel, without a
retained same-thread Sync/Wait explaining it. Rank6 pack1/2 gaps were34.784126
and24.638541ms, overlapping stream synchronizations34.411438/24.300260ms.
The larger gap can include host work, queued or cross-stream dependencies,
or missing/ambiguous positive launch linkage;
it is NOT established as encoder arithmetic. Linked GPU endpoints do not
guarantee coverage of every graph or producer operation.

All16 long collective identities appeared at global layer1 or25, the first
layer of each PP stage. Later-microbatch stalls still require predecessor
analysis. Same-microbatch/same-layer routed expert GEMM occurs AFTER routing,
so cannot cause the first layer1 routing stall. This does not establish general
expert balance or exclude preceding expert work in other microbatches.

Provenance hashes (private raw data are not embedded):

- clock-solution.json: b0461b5900424d5381b22db1734c800aa2ff34e0b345c1c182f5eb06f6348119
- producer-chain.json: c10ceff2c095d5c234de7a3c3590ee3054d6745c5014f2611554a606e574a380
- original full report: caa2a5cbe3b45507009b0be12aea42fbc99d64b3e3a1c9483bb08eb61c87841b

Independent primary review passed. The configured second-model service returned
empty content, so no second-model agreement is claimed. No optimization or
controlled intervention was tested; the unresolved866ms path is the next
diagnostic target, not a demonstrated fix.
