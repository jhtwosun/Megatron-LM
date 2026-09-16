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
controlled intervention was tested. The follow-up below narrows the formerly
unresolved866ms path; no optimization fix has been tested.

## Read-only follow-up: first attention preparation call (422830)

Existing traces were analyzed on an allocated CPU node; no new training ran.
The866.145ms interval was not GPU idle: an867.225ms embedding AllGather occupied
the same stream as the subsequent vision GEMM. The host's863.430ms
`cuKernelSetAttribute` interval overlaps downstream waiting; this does not
establish the API's internal blocking mechanism. Pack identity matching locates
the awaited producer on rank1's
previous vision pack. Another883ms interval similarly overlaps stream
synchronization and a previous-pack AllGather awaiting rank13.

Following those dependencies back reveals the slow producers' first vision
attention call, inside `FusedAttnFunc` -> `nvte_flash_attn_fwd`:

| Rank / step / pack | First native call ms | Max of remaining53 calls ms |
|---|---:|---:|
|0 /5 /0|865.420026|0.166273|
|1 /5 /1|817.709461|0.159552|
|13 /6 /1|861.705313|0.243969|
|12 /7 /0|919.782773|0.239521|
|6 /5 /0, fast reference|0.016544|0.154593|

The54 ranges are workspace-query plus execution calls across27 vision layers,
not54 layers or GPU kernels. Position interpolation was31–34ms and vision
RoPE10–12ms in the earliest slow-producer cases, not the approximately0.9s origin.

The retained TE2.18 torch source calls native attention first with an unallocated
workspace, then allocates and executes. The [v2.18 common source](https://github.com/NVIDIA/TransformerEngine/blob/v2.18/transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu)
looks up a thread-local descriptor cache and, on a miss, builds the cuDNN graph
and execution plans before returning the workspace size. This strongly supports
first-use graph/plan preparation for a newly encountered attention descriptor.
It is not yet a directly recorded cache miss or plan-build interval; matching
versioned source alone does not prove loaded binary identity. Changes in image
shape also do not necessarily change a quantized cache key.

Observed propagation: first attention preparation delay -> embedding gather
wait -> next pack same-stream wait -> window-delivery skew -> decoder EP wait.
Direct backend plan-build logging is the remaining confirmation, not another
claim that NCCL bandwidth or encoder arithmetic is intrinsically slow.

Follow-up provenance:

- gap.json:66ec7008970dd2ef560ec8ffa34e173e41a104684d6a91bc69066c12b2723a3c
- pack-chain.json:5e6382798f679dac6f923d235ec2968803d37f86d830b69dab2a971452fa6943
- vision-attention.json:eb1bd46686912b58f29869f57532e73ac5dfa1ff5431e109ecc0c9f0005a7d0d

Allocation422830 ended normally after9:06. Initial read-only query step1 failed
its original/recompute phase-count assertion; the corrected original-sidecar
filter passed in step2. Other analysis steps passed. This is not a new GPU or
portable-wrapper qualification, nor an end-to-end speedup result.
