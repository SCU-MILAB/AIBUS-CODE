#!/usr/bin/env bash
set -euo pipefail
cd /home/lx/baselines/vjepa
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 --nproc_per_node=4 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:29522 --rdzv-id=schedfix_ft_reg2_clip \
  -m distill.det_vjepa.train \
  --data /home/lx/dataset/6cls_detection/data.yaml \
  --init_weights distill/det_vjepa/output/vjepa_giant_fcos_schedfix/best.pt \
  --output_dir distill/det_vjepa/output --exp_name schedfix_ft_reg2_clip_4gpu \
  --imgsz 640 --batch 8 --accum 1 --val_batch 8 \
  --epochs 20 --lr_backbone 1e-6 --lr_head 1e-5 \
  --drop_path 0.1 --layer_decay 0.85 --weight_decay 0.05 \
  --warmup_epochs 0.5 --min_lr_ratio 0.01 --clip_grad 1.0 --reg_weight 2.0 \
  --val_every 1 --val_max_images 0 --final_val_max_images 0 \
  --conf_thres 0.25 --iou_thres 0.7 --save_period 0 --workers 8 --seed 0
