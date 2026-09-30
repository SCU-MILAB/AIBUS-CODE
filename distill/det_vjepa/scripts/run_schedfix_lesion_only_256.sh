#!/usr/bin/env bash
set -euo pipefail
cd /home/lx/baselines/vjepa
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 --nproc_per_node=4 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:29532 --rdzv-id=schedfix_lesion_only_256 \
  -m distill.det_vjepa.train \
  --data /home/lx/dataset/6cls_detection/data-2cls-remap.yaml \
  --init_weights distill/det_vjepa/output/vjepa_giant_fcos_schedfix/best.pt \
  --output_dir distill/det_vjepa/output --exp_name schedfix_lesion_only_256_remap_4gpu \
  --lesion_only --black_threshold 3 \
  --imgsz 256 --batch 8 --accum 1 --val_batch 8 \
  --epochs 20 --lr_backbone 3e-6 --lr_head 5e-5 \
  --drop_path 0.1 --layer_decay 0.85 --weight_decay 0.05 \
  --warmup_epochs 1.0 --min_lr_ratio 0.01 --clip_grad 1.0 --reg_weight 1.0 \
  --val_every 1 --val_max_images 0 --final_val_max_images 0 \
  --conf_thres 0.25 --iou_thres 0.7 --save_period 0 --workers 8 --seed 0
