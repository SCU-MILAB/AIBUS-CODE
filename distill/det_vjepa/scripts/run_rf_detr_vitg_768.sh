#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
cd "$repo_root"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
IFS=',' read -r -a rf_gpus <<< "$CUDA_VISIBLE_DEVICES"
rf_world="${#rf_gpus[@]}"
rf_batch="${BATCH:-8}"
rf_effective="${EFFECTIVE_BATCH:-16}"
if (( rf_effective % (rf_world * rf_batch) != 0 )); then
  echo "EFFECTIVE_BATCH must be divisible by GPU count times BATCH" >&2
  exit 1
fi
rf_accum="$(( rf_effective / (rf_world * rf_batch) ))"
rf_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%S)_$$}"
rf_name="rf_detr_vitg_768_${rf_tag}"
rf_output="$repo_root/distill/det_vjepa/output"
rf_log="$rf_output/${rf_name}.log"
if [[ -e "$rf_output/$rf_name" || -e "$rf_log" ]]; then
  echo "Refusing to overwrite existing experiment $rf_name" >&2
  exit 1
fi
rf_encoder="$repo_root/ckpts/vjepa_full/best_vjepa_model9639(paper).pt"
[[ -f "$rf_encoder" ]]
echo "Starting $rf_name: GPUs=$CUDA_VISIBLE_DEVICES, batch=$rf_batch, accum=$rf_accum, effective batch=$rf_effective"
echo "Training log: $rf_log"
python -m torch.distributed.run --nproc_per_node="$rf_world" \
  --master_port="${MASTER_PORT:-29671}" -m distill.det_vjepa.rf_detr.train \
  --data /home/lx/dataset/6cls_detection/data-2cls-remap.yaml \
  --encoder_checkpoint "$rf_encoder" \
  --output_dir "$rf_output" --exp_name "$rf_name" \
  --imgsz 768 --epochs "${EPOCHS:-12}" \
  --hidden_dim 256 --decoder_layers 3 --num_queries 300 --group_detr 13 \
  --batch "$rf_batch" --accum "$rf_accum" --val_batch 8 --workers 4 \
  --lr_backbone 1e-5 --lr_head 1e-4 --layer_decay 0.85 --weight_decay 0.05 \
  --warmup_epochs 0.5 --min_lr_ratio 0.01 --clip_grad 0.1 --drop_path 0.1 \
  --grad_checkpoint --amp --flip_prob 0.5 --seed 0 --sampler_seed 0 \
  --val_every 1 --conf_thres 0.25 --iou_thres 0.7 --black_threshold 3 \
  "$@" > "$rf_log" 2>&1
