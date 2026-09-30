#!/usr/bin/env bash
# Paired six-epoch experiments; only the regression loss weight differs.
# Usage: bash .../run_schedfix_lesion_reg_ablation.sh [both|reg1|reg2]
# Run when GPUs 0-3 are free. This script does not stop existing jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

mode="${1:-both}"
case "$mode" in
  both) weights=(1 2) ;;
  reg1) weights=(1) ;;
  reg2) weights=(2) ;;
  *) echo 'Usage: run_schedfix_lesion_reg_ablation.sh [both|reg1|reg2]' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo 'Expected at most one argument.' >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#devices[@]} != 4 )); then
  echo 'This experiment requires four visible GPUs (effective batch 32).' >&2
  exit 2
fi
port="${MASTER_PORT:-29541}"
tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%S)_$$}"
data=/home/lx/dataset/6cls_detection/data-2cls-remap.yaml
init=distill/det_vjepa/output/vjepa_giant_fcos_schedfix/best.pt
output=distill/det_vjepa/output
for path in "$data" "$init"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input: $path" >&2
    exit 1
  fi
done

# Preflight all destinations before launching either member of the pair.
for weight in "${weights[@]}"; do
  name="schedfix_lesion_640_remap_reg${weight}_6ep_${tag}"
  if [[ -e "$output/$name" ]]; then
    echo "Output already exists: $output/$name; use a new RUN_TAG." >&2
    exit 1
  fi
done

for weight in "${weights[@]}"; do
  name="schedfix_lesion_640_remap_reg${weight}_6ep_${tag}"
  mkdir "$output/$name"
  echo "Starting $name; GPUs=$CUDA_VISIBLE_DEVICES; reg_weight=$weight"
  torchrun \
    --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="localhost:$port" --rdzv-id="$name" \
    -m distill.det_vjepa.train \
    --data "$data" --init_weights "$init" \
    --output_dir "$output" --exp_name "$name" \
    --lesion_only --black_threshold 3 --selection_metric lesion_AP50_95 \
    --imgsz 640 --batch 8 --accum 1 --val_batch 8 \
    --epochs 6 --lr_backbone 2e-6 --lr_head 2e-5 \
    --drop_path 0.1 --layer_decay 0.85 --weight_decay 0.05 \
    --warmup_epochs 0.5 --min_lr_ratio 0.01 --clip_grad 1.0 \
    --reg_weight "$weight" \
    --val_every 1 --val_max_images 0 --final_val_max_images 0 \
    --conf_thres 0.25 --iou_thres 0.7 --save_period 0 --workers 4 --seed 0 \
    2>&1 | tee "$output/$name/train.log"
done
