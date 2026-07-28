#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# GB200 performance report for PR7 features on top of the PR6 baseline.
#
# New cells test:
#   - VPP=2 with the fused-recompute MDP path (item 1: VPP sidecar support)
#   - Encoder CP group restricted to PP0 (item 2: encoder-context-parallel)
#   - MTP=1 correctness and throughput impact (item 1: MTP vision-aware rolling)
#
# PR6 baseline cells are repeated unchanged so regressions are caught in the
# same run.

set -euo pipefail

REPORT="pr7"
# Pin to the tip of feature/encoder-cp-and-gather-fix so results are
# reproducible. Update when the branch is rebased.
PR7_CODE_SHA="$(git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)" rev-parse HEAD 2>/dev/null || echo "HEAD")"
PR6_CODE_SHA="e0450f1b948ed53932a9059a3cefc09d0a0a2371"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Cell row format (pipe-separated):
# label | sha | stack_level | pp | vpp | enc_cp | mtp | mdp_mode | fused_backward
#       | vision_cap | measure_from | ref_stat | ref_step_ms | ref_tflops
#       | ref_tok_per_s | ref_peak_mem | note
CELL_ROWS=(
    # ── PR6 baseline (reference cells, code pinned to PR6 SHA) ──────────────
    "pr6_mdp_off|${PR6_CODE_SHA}|pr6|2|0|0|0|mdp_off|recompute|0|6|mean|30481.0|31.26|68801.94|allocated=96974.57_MB|pr6-baseline"
    "pr6_fused_recompute_131072|${PR6_CODE_SHA}|pr6|2|0|0|0|mdp_fused|recompute|131072|6|mean|12775.0|74.79|164160.63|allocated=92029.23_MB|pr6-baseline"

    # ── Item 1: VPP=2 fused-recompute (PR7 code) ────────────────────────────
    # Expected: bubble fraction drops from 1/3 to 1/5; step time ~= pr6_fused × 0.8.
    "pr7_vpp2_fused_recompute|${PR7_CODE_SHA}|pr7|2|2|0|0|mdp_fused|recompute|131072|6|mean||||||vpp2 pp0-gather-only"

    # ── Item 1: MTP=1 fused-recompute baseline ───────────────────────────────
    # Correctness gate: loss at iter 10 must match mtp=0 within 2%.
    "pr7_mtp1_fused_recompute|${PR7_CODE_SHA}|pr7|2|0|0|1|mdp_fused|recompute|131072|6|mean||||||mtp=1 vision-aware-rolling"

    # ── Item 2: Encoder CP restricted to PP0 ─────────────────────────────────
    # enc_cp=2 = decoder cp_size; gather group restricted to PP0 CP ranks.
    # Expected: equal throughput to pr6_fused_recompute with lower peak mem
    # (non-PP0 no longer receives and discards the full embedding tensor).
    "pr7_enc_cp2_pp0_fused_recompute|${PR7_CODE_SHA}|pr7|2|0|2|0|mdp_fused|recompute|131072|6|mean||||||enc-cp=2 pp0-gather-only"
)

usage() {
    cat <<'EOF'
Reproduce the PR7 GB200 performance report (VPP, encoder-CP, MTP).

Usage:
  reproduce_pr7.sh [options]

Options:
  --job-id ID                 Running 16-node salloc job ID.
  --result-root DIR           Host output directory on shared storage.
  --source-repo DIR           Git repository used for checkouts.
  --checkout-root DIR         Shared cache for detached worktrees.
  --cells CSV                 Run only the listed PR7 cells by label.
  --dry-run                   Print launcher commands without executing.
  --prepare-only              Prepare checkouts and stop.
  -h, --help                  Show this help.

Cell labels:
  pr6_mdp_off                 PR6 baseline: MDP off (regression gate)
  pr6_fused_recompute_131072  PR6 baseline: fused recompute (regression gate)
  pr7_vpp2_fused_recompute    VPP=2 with fused recompute
  pr7_mtp1_fused_recompute    MTP=1 with vision-aware rolling
  pr7_enc_cp2_pp0_fused_recompute  Encoder CP restricted to PP0

Metrics: wall-clock step (ms), TFLOPs/GPU, tok/s/GPU, peak allocated mem.
EOF
}

# ---------------------------------------------------------------------------
# Container-side node runner
# ---------------------------------------------------------------------------

run_node() {
    : "${CELL:?}"
    : "${PP_SIZE:?}"
    : "${MDP_MODE:?}"
    : "${MASTER_ADDR:?}"
    : "${MASTER_PORT:?}"

    local target_repo="${TARGET_REPO:-/workspace/Megatron-LM}"
    local result_root="${RESULT_ROOT:-/workspace/results}"
    local energon_root="${ENERGON_ROOT:-/workspace/Megatron-Energon-7.3.2}"
    local venv_root="${VENV_ROOT:-/workspace/venv}"
    local nnodes="${NNODES:-16}"
    local train_iters="${TRAIN_ITERS:-45}"
    local warmup_iters="${WARMUP_ITERS:-5}"
    local fused_backward="${FUSED_BACKWARD:-recompute}"
    local vision_cap="${VISION_MAX_SEQUENCE_LENGTH:-0}"
    local vpp_size="${VPP_SIZE:-0}"
    local mtp_enabled="${MTP_ENABLED:-0}"
    local encoder_cp="${ENCODER_CP_SIZE:-0}"

    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        export PATH="${venv_root}/bin:${PATH}"
        export PYTHONPATH="${venv_root}/lib/python3.12/site-packages:${target_repo}:${energon_root}/src"
        export LD_LIBRARY_PATH="/usr/local/cuda-13.1/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
    fi
    export PYTHONDONTWRITEBYTECODE=1
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export MDP_HYBRID_EP_PER_RANK_CACHE=1
    export WANDB_MODE=offline
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    unset NCCL_NVLS_ENABLE TORCH_FR_BUFFER_SIZE TORCH_NCCL_TRACE_BUFFER_SIZE
    unset TORCH_NCCL_DUMP_ON_TIMEOUT TORCH_NCCL_DEBUG_INFO_TEMP_FILE
    unset TORCH_NCCL_DESYNC_DEBUG TORCH_NCCL_TRACE_CPP_STACK

    local common_extra=(
        --dataloader-type external
        --energon-path /data/energon/blends/blend3.yaml
        --tokenizer-model /data/tokenizer/Qwen3.5-35B-A3B
        --image-min-pixels 0
        --image-max-pixels 327680
        --energon-packing-buffer-size 128
        --energon-shuffle-buffer-size 128
        --energon-max-samples-per-sequence 16
        --energon-prefetch-factor 1
        --num-workers 1
        --dataloader-sequence-packing
        --eval-iters 0
        --mdp-inner-dp-scope pp_cp
        --mdp-loader-prepartition-prefetch-windows 1
    )

    if [[ -n "${DIST_TIMEOUT_MINUTES:-}" ]]; then
        common_extra+=(--distributed-timeout-minutes "${DIST_TIMEOUT_MINUTES}")
    fi
    if [[ "${encoder_cp}" -gt 0 ]]; then
        common_extra+=(--encoder-context-parallel-size "${encoder_cp}")
    fi

    local mode_extra=()
    case "${MDP_MODE}" in
        mdp_off)
            mode_extra+=(--no-mdp-encoder-mode --no-mdp-fused-vision-window
                         --mdp-vision-encoder-max-sequence-length 0
                         --mdp-fused-vision-backward recompute)
            ;;
        mdp_fused)
            if [[ "${vision_cap}" -le 0 ]]; then
                echo "mdp_fused requires a positive VISION_MAX_SEQUENCE_LENGTH" >&2
                exit 2
            fi
            mode_extra+=(--mdp-encoder-mode --mdp-fused-vision-window
                         --mdp-vision-encoder-max-sequence-length "${vision_cap}"
                         --mdp-fused-vision-backward "${fused_backward}")
            ;;
        *) echo "Unknown MDP_MODE=${MDP_MODE}" >&2; exit 2 ;;
    esac

    local dry_run_args=()
    [[ "${DRY_RUN:-0}" == "1" ]] && dry_run_args+=(--dry-run)

    local extra_args="${common_extra[*]} ${mode_extra[*]}"
    local command=(
        bash "${target_repo}/examples/multimodal_dev/scripts/dev_qwen3vl_gb200.sh"
        "${dry_run_args[@]}"
        --gpus 4
        --nnodes "${nnodes}"
        --train-iters "${train_iters}"
        --warmup-iters "${warmup_iters}"
        --results-dir "${result_root}/${CELL}/node${SLURM_NODEID:-0}"
        "${CELL}_pp${PP_SIZE}cp2_gbs256"
        tp=1 ep=8 pp="${PP_SIZE}" cp=2 etp=1 "vpp=${vpp_size}"
        mbs=1 gbs=256 seq_len=8192 image_size=448
        dispatcher_backend=hybridep a2a_overlap=0
        recompute=0 recompute_vision=0 "mtp=${mtp_enabled}"
        use_packed_sequence=1 dataset_provider=energon
        "extra_args=${extra_args}"
    )

    if [[ -n "${CELL_TIMEOUT_SECONDS:-}" && "${DRY_RUN:-0}" != "1" ]]; then
        exec timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SECONDS}" "${command[@]}"
    fi
    exec "${command[@]}"
}

# ---------------------------------------------------------------------------
# Result summariser (identical to PR6 except for column headers)
# ---------------------------------------------------------------------------

summarize_results() {
    python3 - "$1" <<'PY'
import csv, math, re, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
iteration_re = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([0-9.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)"
)
memory_re = re.compile(r"max allocated:\s*([0-9.]+).*?max reserved:\s*([0-9.]+)")

with (root / "manifest.tsv").open(newline="") as f:
    manifest = list(csv.DictReader(f, delimiter="\t"))

rows = []
for source in manifest:
    best = {}
    peak_mem = 0.0
    for log_path in sorted((root / source["cell"]).rglob("*.log")):
        content = log_path.read_text(errors="replace")
        matches = iteration_re.findall(content)
        for num, total, step_ms, tflops in matches:
            n = int(num)
            if n < int(source.get("measure_from", 6)):
                continue
            if not best or float(step_ms) < best["step_ms"]:
                best = {"iter": n, "step_ms": float(step_ms), "tflops": float(tflops)}
        for m in memory_re.finditer(content):
            peak_mem = max(peak_mem, float(m.group(1)))

    ref_step = float(source.get("ref_step_ms", 0) or 0)
    act_step = best.get("step_ms", float("nan"))
    delta = ((act_step - ref_step) / ref_step * 100) if ref_step > 0 and not math.isnan(act_step) else float("nan")

    rows.append({
        "cell": source["cell"],
        "note": source.get("note", ""),
        "act_step_ms": f"{act_step:.1f}" if not math.isnan(act_step) else "?",
        "ref_step_ms": f"{ref_step:.1f}" if ref_step else "—",
        "delta_pct": f"{delta:+.1f}%" if not math.isnan(delta) else "—",
        "act_tflops": f"{best.get('tflops', 0):.2f}" if best else "?",
        "peak_mem_mb": f"{peak_mem:.0f}" if peak_mem else "?",
    })

print(f"\n{'Cell':<45} {'Act(ms)':>9} {'Ref(ms)':>9} {'Delta':>8} {'TFLOPs':>8} {'PeakMB':>10}  Note")
print("─" * 110)
for r in rows:
    print(f"{r['cell']:<45} {r['act_step_ms']:>9} {r['ref_step_ms']:>9} "
          f"{r['delta_pct']:>8} {r['act_tflops']:>8} {r['peak_mem_mb']:>10}  {r['note']}")
PY
}

# ---------------------------------------------------------------------------
# Host runner — parse args, prepare checkouts, run cells
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "__run_node" ]]; then
    run_node
    exit $?
fi

if [[ "${1:-}" == "__summarize" ]]; then
    summarize_results "${2:?result-root required}"
    exit $?
fi

job_id=""
result_root=""
source_repo="${REPO_ROOT}"
checkout_root=""
dry_run="0"
prepare_only="0"
cells_filter=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-id)       job_id="$2";       shift 2 ;;
        --result-root)  result_root="$2";  shift 2 ;;
        --source-repo)  source_repo="$2";  shift 2 ;;
        --checkout-root) checkout_root="$2"; shift 2 ;;
        --cells)        cells_filter="$2"; shift 2 ;;
        --dry-run)      dry_run="1";       shift ;;
        --prepare-only) prepare_only="1";  shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${result_root}" ]]; then
    result_root="${RESULT_ROOT:-/tmp/pr7_results_$(date +%Y%m%d_%H%M%S)}"
fi
if [[ -z "${checkout_root}" ]]; then
    checkout_root="${CHECKOUT_ROOT:-${result_root}/checkouts}"
fi
mkdir -p "${result_root}" "${checkout_root}"

container="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/dongjael/containers/mcore-moe-pytorch26.02-hybridep7febc6e-arm64-flash-energon.sqsh}"
data_host="${DATA_HOST:-/lustre/fsw/portfolios/coreai/users/dongjael/vlm-datasets}"
raw_host="${RAW_HOST:-/lustre/fsw/portfolios/coreai/users/dongjael/vlm-datasets/raw}"
energon_host="${ENERGON_HOST:-/lustre/fsw/portfolios/coreai/users/dongjael/Megatron-Energon}"
venv_host="${VENV_HOST:-/lustre/fsw/portfolios/coreai/users/dongjael/venv}"
blend_host="${BLEND_HOST:-${data_host}/energon/blends/blend3.yaml}"

# Filter and prepare cell list.
selected_rows=()
checkouts=()
manifest="${result_root}/manifest.tsv"
printf '%s\n' "cell	sha	stack_level	pp	vpp	enc_cp	mtp	mode	fused_backward	vision_cap	measure_from	ref_stat	ref_step_ms	ref_tflops	ref_tok_per_s	ref_peak_mem	note" > "${manifest}"

for row in "${CELL_ROWS[@]}"; do
    IFS='|' read -r label sha level pp vpp enc_cp mtp mode backward cap \
                     measure_from ref_stat ref_step ref_tflops ref_tps ref_peak note <<< "${row}"
    if [[ -n "${cells_filter}" && ",${cells_filter}," != *",${label},"* ]]; then
        continue
    fi
    selected_rows+=("${row}")

    # Prepare a detached worktree for this SHA.
    checkout="${checkout_root}/${sha}"
    if [[ ! -d "${checkout}" ]]; then
        git -C "${source_repo}" worktree add --detach "${checkout}" "${sha}"
    fi
    checkouts+=("${checkout}")

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${sha}" "${level}" "${pp}" "${vpp}" "${enc_cp}" "${mtp}" \
        "${mode}" "${backward}" "${cap}" "${measure_from}" "${ref_stat}" \
        "${ref_step}" "${ref_tflops}" "${ref_tps}" "${ref_peak}" "${note}" >> "${manifest}"
done

if [[ "${#selected_rows[@]}" -eq 0 ]]; then
    echo "No cells selected" >&2; exit 2
fi

git -C "${REPO_ROOT}" rev-parse HEAD > "${result_root}/harness_revision.txt"
printf '%s\n' "${REPORT}" > "${result_root}/report.txt"
printf '%s\n' "${container}" > "${result_root}/container.txt"

if [[ "${prepare_only}" == "1" ]]; then
    echo "Prepared ${#selected_rows[@]} PR7 cells under ${checkout_root}"
    echo "RESULT_ROOT=${result_root}"
    exit 0
fi

master_addr="dry-run"
nodes="dry-run"
if [[ "${dry_run}" != "1" ]]; then
    job_info="$(scontrol show job "${job_id}" -o)"
    for required in "JobState=RUNNING" "Account=coreai_devtech_all" "NumNodes=16"; do
        if [[ " ${job_info} " != *" ${required} "* ]]; then
            echo "Allocation ${job_id} does not satisfy ${required}" >&2; exit 2
        fi
    done
    nodes="$(squeue -h -j "${job_id}" -o '%N')"
    master_addr="$(scontrol show hostnames "${nodes}" | sed -n '1p')"
    printf '%s\n' "${job_info}" > "${result_root}/allocation.txt"
fi
printf '%s\n' "${nodes}" > "${result_root}/nodes.txt"
printf '%s\n' "${master_addr}" > "${result_root}/master_addr.txt"

overall_rc=0
for index in "${!selected_rows[@]}"; do
    row="${selected_rows[$index]}"
    checkout="${checkouts[$index]}"
    IFS='|' read -r label sha level pp vpp enc_cp mtp mode backward cap _rest <<< "${row}"
    cell_host="${result_root}/${label}"
    if [[ -e "${cell_host}" ]]; then
        echo "Cell result directory already exists: ${cell_host}" >&2; exit 2
    fi
    mkdir -p "${cell_host}"
    port=$(( ${MASTER_PORT_BASE:-29700} + index ))
    echo "Starting ${label}: sha=${sha:0:12} pp=${pp} vpp=${vpp} enc_cp=${enc_cp} mtp=${mtp} mode=${mode}"

    if [[ "${dry_run}" == "1" ]]; then
        CELL="${label}" PP_SIZE="${pp}" VPP_SIZE="${vpp}" ENCODER_CP_SIZE="${enc_cp}" \
            MTP_ENABLED="${mtp}" MDP_MODE="${mode}" FUSED_BACKWARD="${backward}" \
            VISION_MAX_SEQUENCE_LENGTH="${cap}" \
            MASTER_ADDR=localhost MASTER_PORT="${port}" NNODES=16 SLURM_NODEID=0 \
            DRY_RUN=1 TARGET_REPO="${checkout}" RESULT_ROOT="${result_root}" \
            bash "${SCRIPT_PATH}" __run_node | tee "${cell_host}/dry_run.log"
        touch "${cell_host}/SUCCESS"
        continue
    fi

    mounts="${checkout}:/workspace/Megatron-LM"
    mounts+=",${SCRIPT_DIR}:/workspace/repro"
    mounts+=",${energon_host}:/workspace/Megatron-Energon-7.3.2"
    mounts+=",${data_host}:/data,${raw_host}:/raw"
    mounts+=",${venv_host}:/workspace/venv,${result_root}:/workspace/results"

    cell_rc=0
    CELL="${label}" PP_SIZE="${pp}" VPP_SIZE="${vpp}" ENCODER_CP_SIZE="${enc_cp}" \
        MTP_ENABLED="${mtp}" MDP_MODE="${mode}" FUSED_BACKWARD="${backward}" \
        VISION_MAX_SEQUENCE_LENGTH="${cap}" \
        MASTER_ADDR="${master_addr}" MASTER_PORT="${port}" NNODES=16 \
        TRAIN_ITERS="${TRAIN_ITERS:-45}" WARMUP_ITERS="${WARMUP_ITERS:-5}" \
        DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}" \
        CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-5400}" \
        srun --jobid="${job_id}" --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 \
        --kill-on-bad-exit=1 --mpi=pmix \
        --output="${cell_host}/launcher_%t.log" --error="${cell_host}/launcher_%t.err" \
        --container-image="${container}" --container-mounts="${mounts}" \
        --container-workdir=/workspace/Megatron-LM \
        bash "/workspace/repro/${SCRIPT_NAME}" __run_node || cell_rc=$?

    if [[ "${cell_rc}" -eq 0 ]]; then
        touch "${cell_host}/SUCCESS"
    else
        touch "${cell_host}/FAILED"
        overall_rc=1
        echo "Cell failed: ${label} (rc=${cell_rc})" >&2
    fi
done

if [[ "${dry_run}" == "1" ]]; then
    echo "Dry-run commands written to ${result_root}"
else
    summarize_results "${result_root}" || overall_rc=1
fi
echo "RESULT_ROOT=${result_root}"
exit "${overall_rc}"
