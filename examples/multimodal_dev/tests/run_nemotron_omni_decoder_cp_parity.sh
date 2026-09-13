#!/usr/bin/env bash
set -euo pipefail

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
: "${NVRX_SHIM:?set NVRX_SHIM to the host shim directory}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE to the test container}"
: "${LOG_ROOT:?set LOG_ROOT to the host log directory}"
host_shim=$NVRX_SHIM
container_repo=/workspace/megatron-lm
container_shim=/workspace/nvrx_shim
container=$CONTAINER_IMAGE
test_file=examples/multimodal_dev/tests/test_nemotron_omni_decoder_cp_parity.py
harness_file=examples/multimodal_dev/tests/run_nemotron_omni_decoder_cp_parity.sh
log_root=$LOG_ROOT
mkdir -p "$log_root"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$log_root/$stamp.log"

: "${EXPECTED_HEAD:?set EXPECTED_HEAD to the reviewed N0e commit}"
if [[ ! -f "$host_shim/sitecustomize.py" ]]; then
  printf 'Nemotron CP harness requires the allocation NVRx shim at %s.\n' \
    "$host_shim/sitecustomize.py" >&2
  exit 1
fi
allocation_job=$(scontrol show job "$SLURM_JOB_ID" -o)
allocation_job_name=$(sed -n 's/.* JobName=\([^ ]*\).*/\1/p' <<<"$allocation_job")
if [[ "$allocation_job_name" != coreai_devtech_all-megatron:vlm ]]; then
  printf 'Nemotron CP harness requires allocation JobName=%q; job %s reports %q.\n' \
    coreai_devtech_all-megatron:vlm "$SLURM_JOB_ID" "$allocation_job_name" >&2
  exit 1
fi
test "${SLURM_NNODES:-}" = 1
test "$(git -C "$repo" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$repo" status --short)"
sha256sum "$repo/$test_file" "$repo/$harness_file" | tee "$log"
master_addr=127.0.0.1
master_port=$((23000 + SLURM_JOB_ID % 10000))

srun --overlap --nodes=1 --ntasks=4 --ntasks-per-node=4 \
  --container-image="$container" \
  --container-mounts="$repo:$container_repo,$host_shim:$container_shim" \
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
  --container-mounts="$repo:$container_repo,$host_shim:$container_shim" \
  --container-workdir="$container_repo" \
  bash -lc "
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=$container_shim:$container_repo
    python -m py_compile '$test_file'
  "

git -C "$repo" diff --check HEAD^
