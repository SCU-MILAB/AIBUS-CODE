#!/usr/bin/env bash
set -euo pipefail
cd /home/lx/baselines/vjepa

# Continue from H72 model weights with a fresh optimizer/schedule.
# Extra arguments at the end can override scalar training options.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
torchrun --standalone --nproc_per_node=4 -m distill.det_vjepa.train \
  --data /home/lx/dataset/6cls_detection/data-2cls-remap.yaml \
  --init_weights /home/lx/baselines/vjepa/distill/det_vjepa/output/autoresearch_oct3_h72_reg_weight_0p5/best.pt \
  --output_dir /home/lx/baselines/vjepa/distill/det_vjepa/output \
  --exp_name h72_aug_mild_ft \
  --lesion_only --black_threshold 3 --selection_metric lesion_AP50_95 \
  --imgsz 768 --fpn_strides 8 16 32 --fpn_dim 256 --head_convs 4 \
  --fpn_topdown_scale 1.0 --head_train_scope box_projection \
  --no-freeze_backbone --drop_path 0.1 --grad_checkpoint --amp \
  --batch 4 --accum 4 --val_batch 4 --workers 4 \
  --epochs 3 --cosine_epochs 3 --warmup_epochs 0.1 --train_minutes 0 \
  --lr_backbone 1e-6 --lr_head 1e-5 --layer_decay 0.85 \
  --weight_decay 0.05 --min_lr_ratio 0.01 --clip_grad 1.0 \
  --center_radius 1.25 --reg_weight 0.5 --no-reg_centerness_weighted \
  --flip_prob 0.5 --aug_degrees 3 --aug_translate 0.03 --aug_scale 0.10 \
  --aug_brightness 0.08 --aug_contrast 0.15 --aug_gamma 0.15 \
  --aug_noise_std 0.01 --aug_blur_prob 0.10 \
  --val_every 1 --val_max_images 0 --final_val_max_images 0 \
  --conf_thres 0.25 --iou_thres 0.7 --export_epoch_predictions \
  --save_period 0 --seed 1 --sampler_seed 0 --log_every 50 "$@"
