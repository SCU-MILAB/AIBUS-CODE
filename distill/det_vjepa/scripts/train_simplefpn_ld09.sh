#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Continue from H72 control with a conv-only base lateral and fresh optimizer.
# Append scalar CLI options to override this recipe.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
torchrun --standalone --nproc_per_node=4 -m distill.det_vjepa.train \
  --data /home/lx/dataset/6cls_detection/data-2cls-remap.yaml \
  --init_weights distill/det_vjepa/output/h72_hypotheses_20261009_10ep/train/control/best.pt \
  --output_dir distill/det_vjepa/output \
  --exp_name h72_control_simplefpn_convonly_ld09 \
  --imgsz 768 --fpn_strides 8 16 32 --fpn_dim 256 --head_convs 4 \
  --fpn_topdown_scale 1.0 --head_train_scope box_projection --no-freeze_backbone \
  --drop_path 0.1 --grad_checkpoint --amp \
  --batch 16 --accum 2 --val_batch 4 --workers 2 \
  --epochs 10 --cosine_epochs 10 --warmup_epochs 0.1 \
  --lr_backbone 2e-6 --lr_head 2e-5 --layer_decay 0.9 \
  --weight_decay 0.05 --min_lr_ratio 0.01 --clip_grad 1.0 \
  --lesion_only --black_threshold 3 --selection_metric lesion_AP50_95 \
  --center_radius 1.25 --reg_weight 0.5 --no-reg_centerness_weighted \
  --backbone_gradient all --reg_normalization positive --box_residual none \
  --val_every 1 --val_max_images 0 --final_val_max_images 0 \
  --conf_thres 0.25 --iou_thres 0.7 \
  --export_epoch_predictions --save_period 0 --seed 1 --sampler_seed 0 \
  --reset_training_rng "$@"
