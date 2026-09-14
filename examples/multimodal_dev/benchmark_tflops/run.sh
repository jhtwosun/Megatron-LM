#!/usr/bin/env bash
# Invoke once per node inside an allocation/container with the qualified environment.
set -euo pipefail
mode=${1:?usage: run.sh mock|real}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="${ENERGON_OVERLAY:+${ENERGON_OVERLAY}:}${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline
export CUDA_DEVICE_MAX_CONNECTIONS=8 NCCL_GRAPH_REGISTER=0
export NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API=128
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API=128
export NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0 NVTE_BWD_LAYERNORM_SM_MARGIN=0
export MDP_HYBRID_EP_PER_RANK_CACHE=1
unset NVTE_NORM_FWD_USE_CUDNN NVTE_NORM_BWD_USE_CUDNN
unset PR7_GRAPH_MOCK_FIXED PR7_REFERENCE_MOCK_CONFIG NVTE_DEBUG NVTE_DEBUG_LEVEL
: "${RESULTS_DIR:?Set a fresh, shared output directory visible inside the container}"
NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TP=${TP:-1}
PP=${PP:-2}
CP=${CP:-2}
EP=${EP:-8}
GBS=${GBS:-64}
export SLURM_NODEID=${NODE_RANK:-${SLURM_NODEID:-0}}
if (( NNODES > 1 )); then
    : "${MASTER_ADDR:?Set rendezvous hostname/IP reachable from every node}"
fi
extra="--te-rng-tracker --cuda-graph-warmup-steps 2 --dataloader-sequence-packing --mdp-inner-dp-scope pp_cp --mdp-loader-prepartition-prefetch-windows 1 --distributed-timeout-minutes 20 --mdp-encoder-mode --mdp-fused-vision-window --mdp-vision-encoder-max-sequence-length 131072 --mdp-fused-vision-backward retain --thd-static-packing --max-seqlen-per-dp-cp-rank $((16384 / CP)) --manual-gc-interval 10 --lr-warmup-iters 2 --lr-decay-iters 20 --log-memory-interval 1"
case "$mode" in
    mock)
        provider=mock
        export PR7_REFERENCE_MOCK_CONFIG='{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":512,"max_seq_len":4096,"mean_seq_len":2048,"lognormal_sigma":1.1}'
        extra+=" --num-workers 0 --pack-samples-per-item 4 --thd-max-packed-sequences 32"
        ;;
    real)
        provider=energon
        : "${ENERGON_PATH:?Set the native Energon blend YAML path}"
        : "${TOKENIZER_MODEL:?Set the local HF tokenizer directory}"
        # Native launcher splits extra_args on whitespace.
        if [[ "$ENERGON_PATH" =~ [[:space:]] || "$TOKENIZER_MODEL" =~ [[:space:]] ]]; then
            echo "Energon/tokenizer paths must not contain whitespace" >&2
            exit 2
        fi
        extra+=" --dataloader-type external --energon-path ${ENERGON_PATH} --tokenizer-model ${TOKENIZER_MODEL} --image-min-pixels 0 --image-max-pixels 327680 --energon-packing-buffer-size 128 --energon-shuffle-buffer-size 128 --energon-max-samples-per-sequence 16 --energon-prefetch-factor 1 --energon-report-workload-geometry --num-workers 1 --eval-iters 0 --thd-max-packed-sequences 129"
        ;;
    *) echo "Expected mock or real" >&2; exit 2 ;;
esac
opts=()
if [[ ${DRY_RUN:-0} == 1 ]]; then
    opts+=(--dry-run)
else
    : "${SLURM_JOB_ID:?Run GPU workloads inside an allocation}"
fi
cd "$ROOT"
exec bash examples/multimodal_dev/scripts/dev_qwen3vl_gb200.sh "${opts[@]}" \
    --gpus "$GPUS_PER_NODE" --nnodes "$NNODES" --train-iters 18 --warmup-iters 2 \
    --results-dir "$RESULTS_DIR" "tflops_${mode}" \
    tp="$TP" pp="$PP" cp="$CP" ep="$EP" etp=1 vpp=0 mbs=1 gbs="$GBS" seq_len=16384 \
    vision_num_layers=27 dispatcher_backend=hybridep a2a_overlap=0 use_fsdp=0 \
    recompute=0 recompute_vision=0 mtp=0 use_packed_sequence=1 calculate_per_token_loss=1 \
    dataset_provider="$provider" "cuda_graph_scope=attn moe_router moe_preprocess" "extra_args=$extra"
