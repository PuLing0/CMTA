#!/bin/bash

# Base CMTA training script for TCGA-LUAD (original CMTA repo).
#
# Notes:
# - This original codebase assumes `batch_size=1` (one bag per step).
# - Multi-batch / multi-gpu features exist only in your modified repo.
#
# Examples:
#   # Env style
#   CUDA_VISIBLE_DEVICES=0 \
#   BATCH_SIZE=1 \
#   OOM=8192 \
#   SEED=1 \
#   NUM_EPOCH=150 \
#   bash scripts/luad_ablation_base.sh
#
#   # Script arg style (overrides SEED env)
#   bash scripts/luad_ablation_base.sh --seed 7

set -euo pipefail

PROJ_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJ_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
OOM=${OOM:-4096}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-/home/lukehuang/disk1T/sijuzheng_data/features_dir}
WHICH_SPLITS=${WHICH_SPLITS:-5foldcv}

MODEL=${MODEL:-cmta}
MODEL_SIZE=${MODEL_SIZE:-small}
MODAL=${MODAL:-coattn}
FUSION=${FUSION:-concat}

NUM_EPOCH=${NUM_EPOCH:-150}
LOSS=${LOSS:-nll_surv_l1}
LR=${LR:-0.001}
OPTIMIZER=${OPTIMIZER:-SGD}
SCHEDULER=${SCHEDULER:-None}
ALPHA=${ALPHA:-0.0001}
SEED=${SEED:-1}

# Allow overriding SEED via script args, while forwarding all other args to python.
PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    *)
      PY_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "${BATCH_SIZE}" != "1" ]; then
  echo "[ERROR] This repo version only supports BATCH_SIZE=1 (got BATCH_SIZE=${BATCH_SIZE})." >&2
  echo "        Use your multi-batch CMTA repo if you need BATCH_SIZE>1." >&2
  exit 1
fi

ARGS=(
  --which_splits "${WHICH_SPLITS}"
  --dataset tcga_luad
  --data_root_dir "${DATA_ROOT_DIR}"
  --modal "${MODAL}"
  --model "${MODEL}"
  --model_size "${MODEL_SIZE}"
  --fusion "${FUSION}"
  --num_epoch "${NUM_EPOCH}"
  --batch_size "${BATCH_SIZE}"
  --loss "${LOSS}"
  --lr "${LR}"
  --optimizer "${OPTIMIZER}"
  --scheduler "${SCHEDULER}"
  --alpha "${ALPHA}"
  --seed "${SEED}"
  --OOM "${OOM}"
)

echo "Running CMTA LUAD: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} seed=${SEED} batch_size=${BATCH_SIZE} OOM=${OOM}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python main.py "${ARGS[@]}" "${PY_ARGS[@]}"
