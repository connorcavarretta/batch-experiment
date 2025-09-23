#!/usr/bin/env bash
set -euo pipefail

# GPUs / data root
export CUDA_VISIBLE_DEVICES=0,1,2
export IMAGENET_PATH=/home/hongshen/work/data/ILSVRC/Data/CLS-LOC
export PYTHONUNBUFFERED=1

# NCCL: P2P caused hangs on this machine
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_COLLNET_ENABLE=0
# If socket selection ever causes issues, uncomment:
# export NCCL_SOCKET_IFNAME=eno2np1

# File descriptors (lots of files on ImageNet)
ulimit -n 65536 || true

# Train
python run_batch.py \
  --data  /home/hongshen/work/data/ILSVRC/Data/CLS-LOC \
  --epochs 1 \
  --warmup_epochs 0 \
  --batch_size 64 \
  --base_lr 4e-3 \
  --out ./wpt_ablation \
  --main_py ./main_wpt.py \
  --seed 42 \