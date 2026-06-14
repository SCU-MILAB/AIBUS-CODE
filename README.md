<div align="center">
<h1>BUVFM — Breast Ultrasound Video Foundation Model</h1>
<a href='https://huggingface.co/collections/SCU-MILAB/buvfm'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue'></a>
<a href='https://scu-milab.github.io/BUVFM/'><img src='https://img.shields.io/badge/Homepage-green' alt='Homepage'></a>
<a href='http://buvfm.machineilab.org/'><img src='https://img.shields.io/badge/Demopage-blue' alt='Demopage'></a>
<a href='https://github.com/SCU-MILAB/BUVFM-CODE'><img src='https://img.shields.io/badge/Github-red' alt='Github'></a>
</div>

BUVFM is the first breast ultrasound video foundation model, pretrained on 423K frames and 84K video sequences via self-supervised learning and fine-tuned for BI-RADS risk stratification. It powers AIBS, a cloud-edge system that enables non-physician ultrasound acquisition with real-time quality control, and achieves 95.42% macro sensitivity and 95.60% macro specificity across internal and 11 external cohorts.


- [x] **VJEPA fine-tuning** — Downstream BI-RADS classification on ultrasound videos with ViT-Giant backbone.
- [x] **Inference & evaluation** — Multi-GPU distributed inference, t-SNE/PCA visualization, speckle noise robustness testing.
- [x] **QC screening pipeline** — MobileNetV3 + YOLO frame-level lesion screening and clip extraction.
- [x] **2-Stage demo** — Gradio web app: Stage 1 (MobileNetV3+YOLO screening) → Stage 2 (VJEPA2+Grad-CAM risk grading).
- [x] **Hugging Face Hub release** — Pretrained weights, fine-tuned checkpoints, and QC models on [Hugging Face](https://huggingface.co/xenosscu/BUVFM).
- [x] **Checkpoint conversion** — `pretraining/convert_checkpoint.py` converts MAE/VAE checkpoints to `build_model()`-compatible format for downstream fine-tuning.
- [x] **BUVFM pretraining** — Pretrain a unified BUVFM foundation model on large-scale ultrasound video data.
- [x] **Knowledge distillation** — Distill the ViT-Giant model to smaller architectures for efficient deployment.

## Quick Start
<details>
<summary>Show/Hide</summary>
    
### 1. Install dependencies

```bash
conda create -n buvfm python=3.10.9 -y
conda activate buvfm
pip install -r requirements.txt
```

> **Typical install time**: ~5–10 minutes on a normal desktop computer with broadband internet. Model weight download time (step 2) depends on network speed; weights total approximately 20.9 GB.

### 2. Download demo datasets & checkpoints

```bash
HF_TOKEN=your-huggingface-token python download_dataset_ckpt.py
```

Downloads all demo datasets to `./dataset/` and all model weights to `./` from [xenosscu/BUVFM](https://huggingface.co/xenosscu/BUVFM) on Hugging Face Hub.

#### Datasets

| Name | Save Path | Description |
|------|-----------|-------------|
| `orilong` | `dataset/orilong/` | Original long ultrasound videos |
| `videos_train` | `dataset/videos_train/` | Training set (ImageFolder: `class_0/1/2/`) |
| `videos_val` | `dataset/videos_val/` | Validation set (ImageFolder: `class_0/1/2/`) |

#### Checkpoints

| Save Path | Description |
|-----------|-------------|
| `ckpts/vjepa-nmode-pretrain.pt` | Pretrained ViT-Giant backbone (standard fine-tuning start point) |
| `ckpts/vjepa-nmode-pretrain-vitl.pt` | Pretrained ViT-Large backbone |
| `ckpts/vjepa-nmode-pretrain-vith.pt` | Pretrained ViT-Huge backbone |
| `ckpts/vjepa_full/best_vjepa_model9639(paper).pt` | Paper-reported fine-tuned ViT-Giant checkpoint |
| `ckpts/vjepa_full_vitl/best_vjepa_model.pt` | Fine-tuned ViT-Large checkpoint |
| `ckpts/vjepa_full_vith/best_vjepa_model.pt` | Fine-tuned ViT-Huge checkpoint |
| `distill/cls/output/ckpts/best_distill.pt` | Classification distillation — feature alignment (Stage 1) |
| `distill/cls/output/ckpts/best_finetune_distilled.pt` | Classification distillation — fine-tuned from distilled (Stage 2) |
| `distill/cls/output/ckpts/best_finetune_scratch.pt` | Classification distillation — fine-tuned from scratch baseline |
| `distill/det/output/ckpts/distilled.pt` | Detection distillation — YOLO feature alignment (Stage 1) |
| `distill/det/output/ckpts/ft-scratch.pt` | Detection distillation — YOLO fine-tuned from scratch baseline |

### 3. Prepare your customized dataset

Organize videos in an `ImageFolder`-style directory — each subfolder named `class_N` holds videos of that class:

```
dataset/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── class_2/
└── val/
    ├── class_0/
    ├── class_1/
    └── class_2/
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`. See [Dataset Preparation](#dataset-preparation) for parameter details.

### 4. Fine-tune the pretrained model

```bash
torchrun --master_port=29511 --nproc_per_node=2 train.py \
  --train_dir /path/to/dataset/train \
  --val_dir /path/to/dataset/val \
  --pretrained_ckpt ./ckpts/vjepa-nmode-pretrain.pt \
  --exp_name exp1
```

See [Training](#training) for all arguments (freeze backbone, learning rate, model variants, etc.).

### 5. Run inference

```bash
export TRPORT=29599
export CKPT=/path/to/ckpt
export VAL_DIR='/path/to/val/dir'
export OUT='test'
export EXTRA=""
export FS=2
export FPC=16
torchrun --master_port=$TRPORT --nproc_per_node=4 inference_ddp_old.py \
    --val_dir $VAL_DIR --checkpoint $CKPT --output ./output/$OUT.csv \
    $EXTRA --frame_step $FS --frames_per_clip $FPC
```

See [Inference & Evaluation](#inference--evaluation) for more options.


</details>

## Minimal Reproduction Steps
<details>
<summary>Show/Hide</summary>

Minimal Reproduction Steps to reproduce the train/inference pipeline in our paper.
### Task 1 : Source code、demo dataset and model weight Downloading / Prepare Environment
```bash
# ============================================================
# Step 1. Download source code
# ============================================================
git clone https://github.com/2512988161/BUVFM.git && cd BUVFM/

# ============================================================
# Step 2. Download demo dataset & model weight from HuggingFace
# ============================================================
export HF_ENDPOINT=https://hf-mirror.com
HF_TOKEN=your_huggingface_api_key python download_dataset_ckpt.py 

# ============================================================
# Step 3. Prepare Environment
# ============================================================
conda create -n buvfm python=3.10.9 -y
conda activate buvfm
pip install -r requirements.txt
```
### Task 2: BUVFM Pretraining Full Pipeline
```bash
# ============================================================
# Step 1. Enter pretraining directory
# ============================================================
cd pretraining/

# ============================================================
# Step 2. Download V-JEPA2 ViT-Giant pretrained weights
#   Place under configs/vitg/ as required by prepare.py
# ============================================================
mkdir -p configs/vitg
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-vitg16-256-c4800.pt -O configs/vitg/vitg.pt

# ============================================================
# Step 3. Generate CSV manifests, YAML configs, and launch scripts
#   Outputs: configs/vitg/{pretrain.csv, train.csv, val.csv,
#            pretrain.yaml, finetune.yaml, pretrain.sh, finetune.sh}
# ============================================================
python prepare.py --model vitg \
    --pretrain_dataset ../dataset/videos_train \
    --finetune_dataset_train ../dataset/videos_train \
    --finetune_dataset_val ../dataset/videos_val

# ============================================================
# Step 4. Self-supervised pretraining (V-JEPA2 with nmODE)
#   Checkpoint saved to: outputs/vitg/pretrain/latest.pt
# ============================================================
bash configs/vitg/pretrain.sh

# ============================================================
# Step 5. Fine-tune on video classification (frozen evaluation)
#   Uses pretrain output (outputs/vitg/pretrain/latest.pt) as
#   starting point, configured automatically in finetune.yaml
# ============================================================
bash configs/vitg/finetune.sh

cd ..
```
### Task 3: BUVFM Risk Stratification Fine-tune Pipeline
```bash
# ============================================================
# Step 1. Fine-tune the pretrained model on BI-RADS classification
#   Uses the downloaded nmODE-pretrained ViT-Giant checkpoint
#   Output checkpoint: ./ckpts/vjepa_full/best_vjepa_model.pt
#   Output log: ./output/logs_vjepa/vjepa_full.log
# optional weights:
#    Task 2 output
#    ./pretraining/outputs/vitg/pretrain/latest.pt 
#    ./pretraining/outputs/vitg/finetune/video_classification_finetuned/ultrasound_vjepa2_ALL_finetune/latest.pt
#    BUVFM: 
#     ./ckpts/vjepa-nmode-pretrain.pt
#     ./ckpts/vjepa_full/best_vjepa_model9639(paper).pt
# ============================================================
torchrun --nproc_per_node=4 train.py \
    --pretrained_ckpt ./ckpts/vjepa-nmode-pretrain.pt \
    --train_dir ./dataset/videos_train \
    --val_dir ./dataset/videos_val \
    --batch_size 16 \
    --lr 5e-5 \
    --exp_name reproduction \
    --epochs 50

# ============================================================
# Step 2. Evaluate on validation set
#   Output: ./output/results.csv (per-video predictions)
# ============================================================
export CKPT=$(ls -1 ./ckpts/vjepa_full_reproduction/best_vjepa_model* | sort | tail -n 1)
torchrun --master_port=29599 --nproc_per_node=4 inference_ddp_old.py \
    --val_dir ./dataset/videos_val \
    --checkpoint $CKPT \
    --output ./output/results.csv \
    --frame_step 2 --frames_per_clip 16
```
### Task 4: QC Models Distilling Full Pipeline
```bash
# ============================================================
# Step 1. CLS Distillation — Feature alignment (ViT-G → MobileNetV3)
#   Teacher: paper-reported fine-tuned ViT-Giant checkpoint
#   Student: MobileNetV3-Small backbone
#   Output: distill/cls/output/ckpts/best_distill.pt
# ============================================================
export CKPT=$(ls -1 ./ckpts/vjepa_full_reproduction/best_vjepa_model* | sort | tail -n 1)
# or using BUVFM './ckpts/vjepa_full/best_vjepa_model9639(paper).pt'
torchrun --nproc_per_node=4 distill/cls/distill.py \
    --teacher_ckpt $CKPT \
    --video_dir ./dataset/videos_train \
    --batch_size 40 --epochs 50

# ============================================================
# Step 2. CLS Fine-tuning — 2D image classification
#   Requires a 2-class ImageFolder dataset (class_0/class_1 with images)
#   distill/cls/output/ckpts/best_distill.pt or distill/cls/output/distill/best_distill.pt  
#   Output: distill/cls/output/ft-distill/best_finetune_distilled.pt
# ============================================================
export clsdataroot=/path/to/2cls_classification
torchrun --nproc_per_node=4 distill/cls/finetune.py \
    --distilled_ckpt distill/cls/output/distill/best_distill.pt \
    --train_dir $clsdataroot/train \
    --output_dir distill/cls/output/ft-distill \
    --epochs 200 --batch_size 512
export CLSCKPT=$(ls -1 ./distill/cls/output/ft-distill/best_finetune* | sort | tail -n 1)
python distill/cls/infer.py --model_path $CLSCKPT --data_dir $clsdataroot/val --output_csv distill/cls/output/predictions.csv

# ============================================================
# Step 3. DET Distillation — Feature alignment (ViT-G → YOLO11m)
#   Requires YOLO-format dataset with data.yaml
#   Output: runs/detect/distill/train/
# ============================================================
PYTHONPATH=$(pwd) python distill/det/train.py \
    --teacher_ckpt $CKPT \
    --data /path/to/data.yaml \
    --model yolo11m.pt --imgsz 256 --batch 256 --epochs 100 \
    --output_dir distill/det/output/distill \
    --device 0,1,2,3

export modelpath=runs/detect/distill/train/weights/best.pt
# or using our : distill/det/output/ckpts/distilled.pt
export infername="distilled"
export detdataroot=/root/path/to/yolo/dataset
python distill/det/infer.py --model_path $modelpath  --output distill/det/output/infer/${infername}  --data_dir $detdataroot/images/val2017/  --label_dir $detdataroot/labels/val2017/ --conf 0.25 --imgsz 640
```
### Task 5: QC Full Pipeline
```bash
# ============================================================
# Step 1. Configure paths and runtime environment
#  $clipsave: Directory for saving quality-controlled video clips
#  $inputfolder: Input video directory
#  $CKPT: Pretrained BUVFM checkpoint used for downstream inference
# ============================================================
export clipsave=./output/QC/videos_full
export inputfolder=./dataset/videos_full
export CKPT='./ckpts/vjepa_full/best_vjepa_model9639(paper).pt'
export TRPORT=29500
export OUT=${clipsave}/pred.csv

# ============================================================
# Step 2. Quality Control (QC)
# Perform automatic quality assessment:
#   1. YOLO-based ultrasound region detection
#   2. SVM-based image quality classification
#   3. Save filtered clips for foundation model inference
# Outputs:
#   - QC-selected video clips: ${clipsave}
#   - QC report: out.json
# ============================================================
python distill/quality_con.py --input_folder $inputfolder --yolo_model_path distill/det/output/ckpts/distilled.pt     --svm_model_path distill/cls/output/ckpts/best_finetune_distilled.pt --num_gpus 4 --batch_size 16 --clip_save_dir $clipsave --output_json_name ${clipsave}/out.json

# ============================================================
# Step 3. BUVFM Inference
# Run distributed inference using 4 GPUs.
# Outputs clip-level predictions to pred.csv.
# ============================================================
torchrun --master_port=$TRPORT --nproc_per_node=4 test.py --val_dir $clipsave --checkpoint $CKPT --output $OUT

# ============================================================
# Step 4. Case-level Aggregation
# Merge clip-level predictions into case-level predictions.
# Output: pred_merge.csv
# ============================================================
python merge.py --input_csv ${clipsave}/pred.csv --output_csv ${clipsave}/pred_merge.csv
```

</details>

## Project Structure
<details>
<summary>Show/Hide</summary>

```
BUVFM/
├── download_dataset_ckpt.py    # Download all datasets & model weights from Hugging Face Hub
├── train.py                  # Fine-tuning script
├── inference_ddp_old.py      # Multi-GPU distributed inference 
├── test.py                   # Multi-GPU distributed inference (recursive dir, no labels)
├── app.py                    # Gradio 2-stage pipeline demo (port 9530)
├── pipeline_utils.py         # 2-stage pipeline: MobileNet+YOLO + VJEPA2+Grad-CAM
├── buildmodel.py             # Model construction & checkpoint loading
├── mydataset.py              # VideoFolderDataset, InferenceVideoDataset, etc.
├── utils.py                  # Data augmentation & video transforms
├── cluster_videos_nocohen.py # t-SNE / PCA feature embedding visualization
├── src/
│   ├── models/
│   │   ├── vision_transformer.py   # ViT backbone (giant/large/huge/tiny/...)
│   │   ├── attentive_pooler.py     # Attentive classifier head
│   │   ├── ac_predictor.py         # Action-conditioned predictor
│   │   └── utils/                  # modules, patch_embed, pos_embs
│   ├── masks/                      # Tube/block masking for VJEPA pretraining
│   ├── datasets/                   # Video dataset utils & transforms
│   │   └── utils/
│   │       ├── video/              # RandAugment, RandErase, volume transforms
│   │       ├── weighted_sampler.py
│   │       ├── worker_init_fn.py
│   │       └── dataloader.py
│   └── utils/                      # Distributed, logging, schedulers, checkpoint loader
├── pretraining/
│   ├── prepare.py              # Auto-generate CSVs, YAML configs, and shell scripts
│   ├── app/                    # Pretraining entry points & VJEPA trainer
│   │   ├── main.py             # Single-GPU pretraining entry
│   │   ├── main_distributed.py # Multi-GPU distributed pretraining entry
│   │   ├── scaffold.py         # Config loading & trainer construction
│   │   ├── vjepa/              # VJEPA pretraining engine
│   │   └── vjepa_droid/        # VJEPA-DROID pretraining engine
│   ├── evals/                  # Fine-tuning & evaluation entry points
│   │   ├── main.py             # Single-GPU fine-tuning entry
│   │   ├── main_distributed.py # Multi-GPU distributed fine-tuning entry
│   │   ├── scaffold.py         # Eval config loading
│   │   └── video_classification_frozen/  # Frozen backbone evaluation
│   ├── configs/                # YAML templates & per-model configs
│   │   ├── pretrain-tample.yaml  # Pretrain config template
│   │   ├── fintune-tample.yaml   # Fine-tune config template
│   │   ├── vitg/               # (generated by prepare.py) Configs for ViT-Giant
│   │   ├── vith/               # (generated by prepare.py) Configs for ViT-Huge
│   │   └── vitl/               # (generated by prepare.py) Configs for ViT-Large
│   ├── src/                    # Shared models, datasets, masks, utils
│   └── tests/                  # Unit tests for models and datasets
├── distill/
│   ├── README.md                   # Distillation usage guide
│   ├── cls/                        # Classification distillation (ViT-G → MobileNetV3)
│   │   ├── distill.py              # Stage 1 — feature alignment on videos
│   │   ├── finetune.py             # Stage 2 — 2D image classification fine-tuning
│   │   └── infer.py                # Stage 3 — inference → CSV
│   └── det/                        # Detection distillation (ViT-G → YOLO11m)
│       ├── train.py                # Stage 1 — YOLO training + feature alignment
│       ├── finetune.py             # Stage 1.1 — full YOLO fine-tuning
│       └── infer.py                # Stage 2 — inference → JSON
├── ckpts/
├── assets/                         # 16 example videos (4 per category: Class0/1/2/NO)
└── output/                         # Inference result CSVs & t-SNE outputs

```

</details>

## Dataset Preparation
<details>
<summary>Show/Hide</summary>


Organize videos in an `ImageFolder`-style directory structure — each subfolder named `class_N` holds videos of that class:

```
dataset/
├── train/
│   ├── class_0/
│   │   ├── video_001.mp4
│   │   └── ...
│   ├── class_1/
│   │   └── ...
│   └── class_2/
│       └── ...
└── val/
    ├── class_0/
    ├── class_1/
    └── class_2/
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

Key dataset parameters:

| Parameter | Default | Description |
|---|---|---|
| `frames_per_clip` | 16 | Frames sampled per clip |
| `frame_step` | 2 | Step between sampled frames |
| `num_clips` | 1 | Number of clips per video |
| `random_clip_sampling` | True/False | Random clip start (train) vs. center (val) |

> **Note**: `train.py` applies class-0 undersampling to handle class imbalance (capped at 10,000), and use weighted loss functions.

</details>

## Checkpoint Preparation
<details>
<summary>Show/Hide</summary>

### Download Datasets & Checkpoints from Hugging Face Hub

Use `download_dataset_ckpt.py` to download all demo datasets and model weights (pretrained checkpoint, fine-tuned checkpoints, distillation checkpoints, and QC models) from [xenosscu/BUVFM](https://huggingface.co/xenosscu/BUVFM) on Hugging Face Hub:

```bash
HF_TOKEN=/your/hugging/face/token python download_dataset_ckpt.py
```

This downloads datasets to `./dataset/` and model weights to `./`, preserving the expected directory structure (`ckpts/`, `distill/`, `QC/`).

You can also specify custom output directories or download only datasets/checkpoints:

```bash
# Download only checkpoints to a custom directory
HF_TOKEN=... python download_dataset_ckpt.py --what ckpt --model_dir ./my_models

# Download only datasets to a custom directory
HF_TOKEN=... python download_dataset_ckpt.py --what dataset --dataset_dir ./my_dataset

# Download specific folders only (when --what is dataset or ckpt)
HF_TOKEN=... python download_dataset_ckpt.py --what ckpt --include ckpts/vjepa_full
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--what` | `both` | What to download: `dataset`, `ckpt`, or `both` |
| `--dataset_dir` | `./dataset` | Output directory for datasets |
| `--model_dir` | `.` | Output directory for checkpoints |
| `--include` | (all) | Only download specific folders (prefix matching) |

### Pretrained Starting Point

Fine-tuning starts from `./ckpts/vjepa-nmode-pretrain.pt`. Alternative pretrained checkpoints are available under `ckpts/` (e.g., MAE-pretrained `vitg_mae_e1.pt`, VAE-pretrained `vitg_vae_e1.pt`, ViT-Large `vjepa-vitl.pt`). See the [Pretraining](#pretraining) section for how these are produced.

### Fine-tuned Checkpoints

Fine-tuned checkpoints are saved under `ckpts/vjepa_full/` (or variant directories like `vjepa_full_maee1`, `vjepa_full_newval`, etc.). Naming convention: `best_vjepa_model.pt` (best by validation accuracy).

</details>

## Pretraining
<details>
<summary>Show/Hide</summary>

See `pretraining/README.md` .

</details>

## Training
<details>
<summary>Show/Hide</summary>

All training starts from a pretrained VJEPA checkpoint (`--pretrained_ckpt` is required). The standard starting point is `./ckpts/vjepa-nmode-pretrain.pt`.

### Fine-tuning (`train.py`)

Uses weighted `CrossEntropyLoss` with class-0 undersampling (capped at 10,000 samples).

```bash
torchrun --nproc_per_node=4 train.py \
    --pretrained_ckpt ./ckpts/vjepa-nmode-pretrain.pt \
    --batch_size 16 \
    --lr 5e-5 \
    --epochs 50
```
### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--pretrained_ckpt` | (required) | Path to pretrained checkpoint |
| `--batch_size` | 4 | Batch size per GPU |
| `--lr` | 5e-5 | Learning rate |
| `--epochs` | 50 | Number of training epochs |
| `--num_frames` | 16 | Frames per clip |
| `--model_name` | `vit_giant_xformers` | Backbone: `vit_giant_xformers`, `vit_large`, `vit_huge` |
| `--freeze_backbone` | False | Freeze encoder, train classifier only |
| `--exp_name` | None | Experiment name suffix for log/ckpt directories |
| `--train_dir` | `/home/lx/alg/videos_train` | Training data directory |
| `--val_dir` | `/home/lx/alg/videos_val` | Validation data directory |


### Training Output

- **Logs**: `./output/logs_vjepa/vjepa_full.log` (or `vjepa_frozen.log` if `--freeze_backbone`)
- **Checkpoints**: `./ckpts/vjepa_full/best_vjepa_model.pt` (best by val accuracy)
- **Eval CSV**: `./ckpts/vjepa_full/best_vjepa_eval.csv` (per-video predictions)

</details>

## Knowledge Distillation for Edge-Side Quality Control Model
<details>
<summary>Show/Hide</summary>

See [distill/README.md](distill/README.md) for details.

</details>

## Inference & Evaluation
<details>
<summary>Show/Hide</summary>

### Multi-GPU Distributed Inference (Primary)

Use `inference_ddp_old.py` with `torchrun` for parallel inference across GPUs:

```bash
export TRPORT=29599
export CKPT=/path/to/ckpt
export VAL_DIR='/path/to/val/dir'
export OUT='test'
export EXTRA=""
export FS=2
export FPC=16
torchrun --master_port=$TRPORT --nproc_per_node=4 inference_ddp_old.py \
    --val_dir $VAL_DIR --checkpoint $CKPT --output ./output/$OUT.csv \
    $EXTRA --frame_step $FS --frames_per_clip $FPC
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `./ckpts/vjepa_full/best_vjepa_model.pt` | Path to fine-tuned checkpoint |
| `--val_dir` | `./dataset/videos_val` | One or more validation directories |
| `--output` | `./output/inference_results.csv` | Output CSV path |
| `--model_name` | `vit_giant_xformers` | Backbone variant |
| `--frame_step` | 2 | Step between sampled frames |
| `--frames_per_clip` | 16 | Frames sampled per clip |
| `--restore_true` | False | Run robust evaluation across speckle noise ratios (0.05–0.95) |

Each rank writes a temp CSV; rank 0 merges and deduplicates. Supports resume (skips already-processed videos if the output CSV exists).

### Arbitrary Directory Inference (no labels required)
Recursively scans all videos in the given directory (no `class_N` folder structure required). Output CSV contains `video_name`, `p0`, `p1`, `p2` (no `label` column).

```bash
export TRPORT=29599
export CKPT=/path/to/ckpt
export VAL_DIR='/path/to/val/dir'
export OUT='test'
export EXTRA=""
export FS=2
export FPC=16
torchrun --master_port=$TRPORT --nproc_per_node=4 test.py \
    --val_dir $VAL_DIR --checkpoint $CKPT --output ./output/$OUT.csv \
    $EXTRA --frame_step $FS --frames_per_clip $FPC
```

</details>

## Demo
<details>
<summary>Show/Hide</summary>

### 2-Stage Pipeline Demo

A web-based interactive demo implementing a cascaded 2-stage inference pipeline:

```
Video Input → [Stage 1: Screening] → [Stage 2: Risk Grading]
                    │                         │
          MobileNetV3 + YOLO           VJEPA2 ViT-Giant
          No lesions? → STOP           3-class probs + Grad-CAM
```

Pipeline logic and model interfaces (MobileNetV3, YOLO, VJEPA2, Grad-CAM) are in `pipeline_utils.py`.

#### Running Locally

```bash
conda activate buvfm
python app.py
```

Open http://localhost:9530 in your browser. Upload a video or click an example from `assets/` (16 videos across 4 categories: Class 0 / Class NO / Class 1 / Class 2). The model is loaded on the first pipeline run.

Optional environment variables:

```bash
export BUVFM_CHECKPOINT_PATH=/absolute/path/to/checkpoint.pt
export BUVFM_ASSETS_DIR=/absolute/path/to/assets
export BUVFM_MAX_FILE_SIZE_MB=100
```

#### Expected Output

The demo displays results in a Gradio web UI with two stages:

1. **Stage 1 (Screening)**: Frame-level lesion detection with YOLO bounding boxes and MobileNetV3 classification. Videos with no detected lesions are flagged and the pipeline stops.
2. **Stage 2 (Risk Grading)**: 3-class BI-RADS risk probabilities (Class 0 / Class 1 / Class 2) with Grad-CAM heatmap overlays highlighting diagnostically relevant regions.

Example inference results are saved to `./output/` as CSV files with per-video predictions and confidence scores.

#### Expected Run Time

| Scenario | Time |
|----------|------|
| Model loading (first run, cold start) | ~10–30 seconds |
| Per-video inference (2-stage pipeline) | ~5–15 seconds |
| Total (load + 1 example video) | ~30–60 seconds |

*Times measured on a desktop with NVIDIA RTX 4090 (24 GB VRAM). On CPU-only machines the demo will run significantly slower and is not recommended for production use.*

### Online Demo

A hosted Gradio demo is available at [http://buvfm.machineilab.org/](http://buvfm.machineilab.org/).


</details>

## System Requirements
<details>
<summary>Show/Hide</summary>

### Operating System
| Item | Version |
|------|---------|
| OS | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| Kernel | 6.14.0-37-generic |
| Architecture | x86_64 |

### Hardware

#### Minimum
| Component | Specification |
|-----------|---------------|
| CPU | 8-core x86_64 processor |
| RAM | 32 GB |
| GPU | 1× NVIDIA GPU with 24 GB VRAM (e.g., RTX 4090) |

#### Recommended (this configuration)
| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon Platinum 8568Y+ (192 cores) |
| RAM | 125 GB |
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM each) |

### GPU & CUDA
| Item | Version |
|------|---------|
| GPU Driver | 570.153.02 |
| CUDA | 12.8 |
| cuDNN | Compatible with CUDA 12.x |

### Software Dependencies
| Package | Version |
|---------|---------|
| Python | 3.10.9 |

For all Python package dependencies, see [requirements.txt](requirements.txt).

### Tested Versions
| Item | Tested Version |
|------|----------------|
| Ubuntu | 22.04.5 LTS |
| Linux Kernel | 6.14.0-37-generic |
| Python | 3.10.9 |
| CUDA | 12.8 |
| GPU Driver | 570.153.02 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) |
| PyTorch | ≥ 2.0.0 |

</details>

## License
<details>
<summary>Show/Hide</summary>

This model and associated code are released under the [CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the BUVFM model and its derivatives, which include models trained on outputs from the BUVFM model or datasets created from the BUVFM model, is prohibited and requires prior approval.

</details>
