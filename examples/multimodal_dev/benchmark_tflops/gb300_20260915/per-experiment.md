# Per-experiment recorded-artifact catalog

Descriptive metadata only.231 artifacts cover230 provisional groups:224 accepted groups (225 artifacts), plus6 failed/partial groups. Alternate artifacts are not extra runs. No historical rate has been corrected here; null means unresolved, never zero.

New accepted measurements are listed in the [separate corrected-accounting snapshot](sweep-corrected-snapshot-20260916.md), with proof digests and exact windows. They are additional to this historical inventory, not replacements for its unavailable corrected values.

Machine-readable companion: [historical catalog and original legacy-rate audit](per-experiment.json). Each artifact ID resolves to its original basename, namespace, SHA256, configuration SHA, stack fields and missing-reference list there. The main table prioritizes corrected decoder rates: unavailable means insufficient historical evidence, not zero. Original legacy rates are preserved unchanged in the JSON audit, not presented as corrected values. Private source/log paths are deliberately omitted. Declared topology products and historical window labels are not new scheduler or runtime proof. No pairwise winner is inferred.

**Ambiguous world-size records:** artifact-138/139/140 (EXP-161/162/163) contain configuration products of128; artifact-141/142 (EXP-164/165) contain256. These five values are suspect, scheduler-unvalidated topology products, not established actual GPU counts. They must not be used to normalize rates or compare hardware scales without original-job verification.

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

## Resolution limits

Every record retains original metric semantics and null corrected TFLOPs. Configured image sizes/counts do not establish consumed image or final attended segment geometry. Packing stride is not a document cap. Source pins, missing original logs, ambiguous worlds, and window labels require per-run proof before correction or fair comparison. Failed/partial artifact metrics are null and never included as successful measurements.
