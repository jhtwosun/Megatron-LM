#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

readonly expected_job_name="coreai_devtech_all-megatron:vlm"
readonly expected_parent="2dff78f6b677f47ad040f5ffcc8af5326642ed29"
readonly host_repo="/project/coreai_devtech_all/dongjael/megatron-lm-worktrees/jhtwosun-mdp-e2b-world-integration-20260905"
readonly container_repo="/workspace/megatron-lm-worktrees/jhtwosun-mdp-e2b-world-integration-20260905"
readonly image="/project/coreai_devtech_all/dongjael/mcore-moe-pytorch26.02-hybridep7febc6e-arm64-flash-energon.sqsh"
readonly shim_host="/project/coreai_devtech_all/dongjael/slurm_logs/mdp_pr116_gate7_20260905/nvrx_shim/sitecustomize.py"
readonly test_file="tests/unit_tests/mdp/test_dynamic_cp_d4_encoder_forward_distributed.py"
readonly harness_file="tests/unit_tests/mdp/run_dynamic_cp_d4_encoder_forward_distributed.sh"

: "${EXPECTED_HEAD:?set EXPECTED_HEAD to the reviewed commit}"
: "${EXPECTED_TEST_SHA256:?set EXPECTED_TEST_SHA256 to the reviewed test hash}"
: "${EXPECTED_HARNESS_SHA256:?set EXPECTED_HARNESS_SHA256 to the reviewed harness hash}"
: "${SLURM_JOB_ID:?run inside the held allocation}"

actual_job_name=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.*JobName=\([^ ]*\).*/\1/p')
if [[ "$actual_job_name" != "$expected_job_name" ]]; then
    echo "wrong allocation JobName: expected $expected_job_name, got $actual_job_name" >&2
    exit 2
fi
[[ -f "$shim_host" ]] || { echo "missing NVRx shim: $shim_host" >&2; exit 2; }
[[ $(git -C "$host_repo" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || {
    echo "HEAD does not match EXPECTED_HEAD" >&2
    exit 2
}
[[ $(git -C "$host_repo" rev-parse HEAD^) == "$expected_parent" ]] || {
    echo "reviewed parent mismatch" >&2
    exit 2
}
[[ -z $(git -C "$host_repo" status --porcelain) ]] || {
    echo "worktree must be clean" >&2
    exit 2
}
[[ $(sha256sum "$host_repo/$test_file" | awk '{print $1}') == "$EXPECTED_TEST_SHA256" ]] || {
    echo "test hash mismatch" >&2
    exit 2
}
[[ $(sha256sum "$host_repo/$harness_file" | awk '{print $1}') == "$EXPECTED_HARNESS_SHA256" ]] || {
    echo "harness hash mismatch" >&2
    exit 2
}

world_size=${WORLD_SIZE:-4}
case "$world_size" in
    4) nodes=1 ;;
    8) nodes=2 ;;
    *) echo "WORLD_SIZE must be 4 or 8" >&2; exit 2 ;;
esac
(( SLURM_NNODES >= nodes )) || {
    echo "WORLD_SIZE=$world_size requires $nodes allocated nodes" >&2
    exit 2
}

mapfile -t allocated_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
(( ${#allocated_hosts[@]} == SLURM_NNODES )) || {
    echo "allocation host count does not match SLURM_NNODES" >&2
    exit 2
}
master_addr=${allocated_hosts[0]}
readonly mounts="/project/coreai_devtech_all/dongjael:/workspace,/lustre/fsw/coreai_devtech_all/dongjael:/mnt"

sha256sum "$host_repo/$test_file" "$host_repo/$harness_file"

srun --overlap --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --container-image="$image" --container-mounts="$mounts" \
    --container-workdir="$container_repo" \
    bash -lc '
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONPATH=/workspace/slurm_logs/mdp_pr116_gate7_20260905/nvrx_shim:'"$container_repo"'
        export MASTER_ADDR='"$master_addr"'
        export MASTER_PORT=${MASTER_PORT:-29641}
        python -m torch.distributed.run \
            --nnodes='"$nodes"' --nproc-per-node=4 \
            --node-rank="$SLURM_PROCID" \
            --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
            -m pytest -q '"$test_file"'
    '
