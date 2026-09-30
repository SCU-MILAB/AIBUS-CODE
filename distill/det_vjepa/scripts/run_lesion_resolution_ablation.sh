#!/usr/bin/env bash
# Controlled 640/768 lesion detection comparison. Two concurrent 2-GPU jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

gpus_640="${GPUS_640:-0,1}"
gpus_768="${GPUS_768:-2,3}"
IFS=',' read -r -a devices_640 <<< "$gpus_640"
IFS=',' read -r -a devices_768 <<< "$gpus_768"
if (( ${#devices_640[@]} != 2 || ${#devices_768[@]} != 2 )); then
  echo 'GPUS_640 and GPUS_768 must each contain exactly two GPU indices.' >&2
  exit 2
fi
declare -A selected_gpus=()
for device in "${devices_640[@]}" "${devices_768[@]}"; do
  if [[ ! "$device" =~ ^[0-9]+$ || -v selected_gpus["$device"] ]]; then
    echo 'GPU indices must be nonnegative integers, with no overlap between jobs.' >&2
    exit 2
  fi
  selected_gpus["$device"]=1
done

port="${MASTER_PORT:-29551}"
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65534 )); then
  echo 'MASTER_PORT must be a TCP port in 1..65534 (both port and port+1 are used).' >&2
  exit 2
fi
tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%S)_$$}"
if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo 'RUN_TAG may contain only letters, digits, dots, underscores and hyphens.' >&2
  exit 2
fi

data=/home/lx/dataset/6cls_detection/data-2cls-remap.yaml
init=distill/det_vjepa/output/vjepa_giant_fcos_schedfix/best.pt
output=distill/det_vjepa/output
pair_dir="$output/lesion_resolution_640_vs_768_${tag}"
for path in "$data" "$init"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input: $path" >&2
    exit 1
  fi
done
if [[ -e "$pair_dir" ]]; then
  echo "Output already exists: $pair_dir; use a new RUN_TAG." >&2
  exit 1
fi
# Do this before creating output directories. A stale NVIDIA device node is not
# enough: both jobs require actual CUDA access to their assigned GPU pair.
for gpu_pair in "$gpus_640" "$gpus_768"; do
  if ! CUDA_VISIBLE_DEVICES="$gpu_pair" python -c \
      'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 2 else 1)'; then
    echo "CUDA cannot access both GPUs in pair $gpu_pair; training was not started." >&2
    exit 1
  fi
done
# Each run saves best.pt and the last checkpoint (about 12 GiB each here).
# Leave room for both runs and other processes on the shared filesystem.
available_kib="$(df -Pk "$output" | awk 'NR==2 {print $4}')"
if (( available_kib < 64 * 1024 * 1024 )); then
  echo 'At least 64 GiB of free disk space is needed for both runs.' >&2
  exit 1
fi

mkdir "$pair_dir"
mkdir "$pair_dir/640" "$pair_dir/768"
run_job() {
  local imgsz="$1" gpu_pair="$2" job_port="$3"
  echo "Starting ${imgsz}px; effective batch=32; 6 epochs; GPUs=$gpu_pair; port=$job_port"
  CUDA_VISIBLE_DEVICES="$gpu_pair" torchrun \
    --nnodes=1 --nproc_per_node=2 \
    --rdzv-backend=c10d --rdzv-endpoint="localhost:$job_port" \
    --rdzv-id="lesion_resolution_${tag}_${imgsz}" \
    -m distill.det_vjepa.train \
    --data "$data" --init_weights "$init" \
    --output_dir "$pair_dir" --exp_name "$imgsz" \
    --lesion_only --black_threshold 3 --selection_metric lesion_AP50_95 \
    --imgsz "$imgsz" --batch 4 --accum 4 --val_batch 4 \
    --epochs 6 --lr_backbone 2e-6 --lr_head 2e-5 \
    --drop_path 0.1 --layer_decay 0.85 --weight_decay 0.05 \
    --warmup_epochs 0.5 --min_lr_ratio 0.01 --clip_grad 1.0 \
    --reg_weight 1.0 \
    --val_every 1 --val_max_images 0 --final_val_max_images 0 \
    --conf_thres 0.25 --iou_thres 0.7 --save_period 0 --workers 4 --seed 0 \
    2>&1 | tee "$pair_dir/$imgsz/train.log"
}

run_job 640 "$gpus_640" "$port" &
pid_640=$!
run_job 768 "$gpus_768" "$((port + 1))" &
pid_768=$!

status=0
wait "$pid_640" || status=1
wait "$pid_768" || status=1
if (( status != 0 )); then
  echo "At least one run failed; inspect $pair_dir/{640,768}/train.log." >&2
  exit 1
fi

python -m distill.det_vjepa.scripts.compare_lesion_resolution "$pair_dir"
echo "Comparison saved to $pair_dir/comparison.json"
