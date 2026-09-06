#!/usr/bin/env bash
set -euo pipefail

repo=/project/coreai_devtech_all/dongjael/megatron-lm-worktrees/jhtwosun-mdp-n0e-nemotron-decoder-cp-parity-20260905
container_repo=/workspace/megatron-lm-worktrees/jhtwosun-mdp-n0e-nemotron-decoder-cp-parity-20260905
host_shim=/project/coreai_devtech_all/dongjael/slurm_logs/mdp_pr116_gate7_20260905/nvrx_shim
container_shim=/workspace/slurm_logs/mdp_pr116_gate7_20260905/nvrx_shim
parent=613c716ba5efed26f0af68951279281e9e6d04ea
container=/project/coreai_devtech_all/dongjael/mcore-moe-pytorch26.02-hybridep7febc6e-arm64-flash-energon.sqsh
test_file=examples/multimodal_dev/tests/test_nemotron_omni_decoder_cp_parity.py
harness_file=examples/multimodal_dev/tests/run_nemotron_omni_decoder_cp_parity.sh
log_root=/project/coreai_devtech_all/dongjael/slurm_logs/mdp_n0e_nemotron_decoder_cp_parity_20260905
mkdir -p "$log_root"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$log_root/$stamp.log"

: "${EXPECTED_HEAD:?set EXPECTED_HEAD to the reviewed N0e commit}"
if [[ ! -f "$host_shim/sitecustomize.py" ]]; then
  printf 'N0e harness requires the allocation NVRx shim at %s.\n' \
    "$host_shim/sitecustomize.py" >&2
  exit 1
fi
allocation_job=$(scontrol show job "$SLURM_JOB_ID" -o)
allocation_job_name=$(sed -n 's/.* JobName=\([^ ]*\).*/\1/p' <<<"$allocation_job")
if [[ "$allocation_job_name" != coreai_devtech_all-megatron:vlm ]]; then
  printf 'N0e harness requires allocation JobName=%q; job %s reports %q.\n' \
    coreai_devtech_all-megatron:vlm "$SLURM_JOB_ID" "$allocation_job_name" >&2
  exit 1
fi
test "${SLURM_NNODES:-}" = 1
test "$(git -C "$repo" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$repo" rev-parse HEAD^)" = "$parent"
test -z "$(git -C "$repo" status --short)"
test "$(git -C "$repo" diff-tree --no-commit-id --name-only -r HEAD | sort)" = "$(
  printf '%s\n' \
    "$harness_file" \
    "$test_file" | sort
)"
sha256sum "$repo/$test_file" "$repo/$harness_file" | tee "$log"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
master_port=$((23000 + SLURM_JOB_ID % 10000))

srun --overlap --nodes=1 --ntasks=4 --ntasks-per-node=4 \
  --container-image="$container" \
  --container-mounts=/project/coreai_devtech_all/dongjael:/workspace \
  --container-workdir="$container_repo" \
  bash -lc "
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=$container_shim:$container_repo
    export MASTER_ADDR=$master_addr MASTER_PORT=$master_port WORLD_SIZE=4
    export RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID
    export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0
    python -m pytest -q '$test_file'
  " 2>&1 | tee -a "$log"

srun --overlap --nodes=1 --ntasks=1 \
  --container-image="$container" \
  --container-mounts=/project/coreai_devtech_all/dongjael:/workspace \
  --container-workdir="$container_repo" \
  bash -lc "
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=$container_shim:$container_repo
    python -m py_compile '$test_file'
  "

git -C "$repo" diff --check HEAD^
