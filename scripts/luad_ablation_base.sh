#!/bin/bash

# Base CMTA training script for TCGA-LUAD (original CMTA repo).
#
# Notes:
# - This repo supports true multi-batch training via length-bucketed batching.
#
# Full example (all supported env vars):
#   cd /home/sijuzheng/project/CMTA && \
#   CUDA_VISIBLE_DEVICES=0 \
#   DATA_ROOT_DIR=/home/lukehuang/disk1T/sijuzheng_data/features_dir \
#   WHICH_SPLITS=5foldcv \
#   FOLD=32 \
#   MODEL=cmta \
#   MODEL_SIZE=small \
#   MODAL=coattn \
#   FUSION=concat \
#   NUM_EPOCH=150 \
#   BATCH_SIZE=1 \
#   LOSS=nll_surv_l1 \
#   LR=0.001 \
#   OPTIMIZER=SGD \
#   SCHEDULER=None \
#   ALPHA=0.0001 \
#   SEED=7 \
#   EARLYSTOP_START=20 \
#   PATIENCE=30 \
#   OOM=8192 \
#   NUM_WORKERS=8 \
#   PREFETCH_FACTOR=4 \
#   PIN_MEMORY=1 \
#   bash scripts/luad_ablation_base.sh

set -euo pipefail

PROJ_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJ_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
OOM=${OOM:-4096}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-/home/lukehuang/disk1T/sijuzheng_data/features_dir}
WHICH_SPLITS=${WHICH_SPLITS:-5foldcv}
FOLD=${FOLD:-}

NUM_WORKERS=${NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-0}

EARLYSTOP_START=${EARLYSTOP_START:-0}
PATIENCE=${PATIENCE:-0}

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

ARGS=(
  --which_splits "${WHICH_SPLITS}"
  --fold "${FOLD}"
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
  --earlystop_start "${EARLYSTOP_START}"
  --patience "${PATIENCE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --pin_memory "${PIN_MEMORY}"
  --OOM "${OOM}"
)

echo "Running CMTA LUAD: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} seed=${SEED} fold=${FOLD:-all} batch_size=${BATCH_SIZE} OOM=${OOM} num_workers=${NUM_WORKERS} prefetch_factor=${PREFETCH_FACTOR} pin_memory=${PIN_MEMORY} earlystop_start=${EARLYSTOP_START} patience=${PATIENCE}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python main.py "${ARGS[@]}" "$@"
