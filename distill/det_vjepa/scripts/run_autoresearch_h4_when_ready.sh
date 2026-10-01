#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

data=/home/lx/dataset/6cls_detection/data-2cls-remap.yaml
weights=distill/det_vjepa/output/lesion_resolution_640_vs_768_20260928T083805_1134/768/best.pt
output=distill/det_vjepa/output
exp=autoresearch_sep30_h4_headonly_cos2
log="$output/$exp.log"

gpu_ready() {
    local gpu_free
    gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null) || return 1
    mapfile -t free_mb <<< "$gpu_free"
    [[ ${#free_mb[@]} -eq 4 ]] || return 1
    for amount in "${free_mb[@]}"; do
        [[ $amount =~ ^[0-9]+$ ]] && (( amount >= 16000 )) || return 1
    done
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -c 'import torch; assert torch.cuda.device_count() == 4' >/dev/null 2>&1
}

while ! gpu_ready; do
    printf '%s waiting for four usable GPUs\n' "$(date -u '+%F %T UTC')"
    sleep 50
done

test -f "$data"
test -f "$weights"
test ! -e "$output/$exp"
test ! -e "$log"
available_gb=$(df --output=avail -BG /home | tail -n 1 | tr -d ' G')
(( available_gb >= 20 ))

printf '%s starting %s\n' "$(date -u '+%F %T UTC')" "$exp"
exec env CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=12465 --nproc_per_node=4 \
  -m distill.det_vjepa.train \
  --data "$data" --init_weights "$weights" \
  --output_dir "$output" --exp_name "$exp" \
  --lesion_only --black_threshold 3 --selection_metric lesion_AP50_95 \
  --imgsz 768 --batch 4 --accum 4 --val_batch 4 \
  --epochs 100 --cosine_epochs 2 --lr_backbone 2e-6 --lr_head 2e-5 \
  --drop_path 0.1 --layer_decay 0.85 --weight_decay 0.05 \
  --warmup_epochs 0.1 --min_lr_ratio 0.01 --clip_grad 1.0 --reg_weight 1.0 \
  --val_every 1 --val_max_images 0 --final_val_max_images 0 \
  --conf_thres 0.25 --iou_thres 0.7 --save_period 0 --workers 4 --seed 0 \
  --train_minutes 120 --save_min_score 0.8104124888777733 \
  --freeze_backbone > "$log" 2>&1
