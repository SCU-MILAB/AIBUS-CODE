# ViT-G + RF-DETR comparison

This isolated module adapts the official Roboflow RF-DETR detector to the existing
V-JEPA ViT-G encoder. It does not replace the project's FCOS trainer or evaluator.

## Architecture and initialization

- Upstream: https://github.com/roboflow/rf-detr, tag `1.9.4`, commit
  `9b009fa928d6218320439803d1da01869a85c072`.
- `vendor/rfdetr` contains the official projector, transformer, deformable
  attention, LWDETR, Hungarian matching, and loss implementations. Algorithm files
  only change their import namespace. `vendor/provenance.json` records source and
  adapted hashes; `vendor/LICENSE` contains the Apache 2.0 license.
- The original DINO backbone builder is disabled. `VJepaRFBackbone` calls the
  project's existing `build_vjepa_backbone` and strictly initializes its encoder
  from `ckpts/vjepa_full/best_vjepa_model9639(paper).pt`.
- The input is a fixed 768 × 768 letterboxed image, repeated into two frames as in
  the existing detector. The final 48 × 48 ViT-G feature map feeds the official
  P4 projector. The backbone is trainable with activation checkpointing.
- Detection defaults: hidden dimension 256, three decoder layers, 300 queries,
  13 training query groups, two-stage proposals, two sampling points, lightweight
  reference-point refinement and box reparameterization. Training produces 3,900
  queries; inference produces 300. These follow the RF-DETR Base head design.
- The RF projector and detector start from random initialization. No COCO detector
  weights, DINO weights, H72 detection weights, or teacher are loaded.
- Only lesion label 0 is trained. The upstream extra logit is retained for its
  native loss convention; it is not emitted as a background detection.
- Matching costs are class/L1/GIoU = 2/5/2. Loss weights are 1/5/2, with the
  official IoU-aware BCE, decoder auxiliary losses and encoder losses.

This is an RF-DETR comparison with a substituted ViT-G backbone and the project's
common evaluation protocol. It is not a stock pretrained RF-DETR benchmark.

## Evaluation contract

`evaluation.py`, `metric.py`, `dataset.py`, and `distributed.py` are reused without
edits. Their pre-integration hashes are in `protected_files_sha256.json`.

The adapter converts RF-DETR normalized boxes to input-pixel `xyxy` and emits the
lesion sigmoid score. All confidence filtering (`0.25`), NMS (`0.7`), restoration
to original coordinates, clipping, background completion, ground truth and metric
calculation remain in the existing evaluator. RF-DETR has no centerness branch;
its score is its lesion sigmoid. The evaluator's legacy background metadata text
is retained for output compatibility; the RF run config records the actual score
rule.

The original full validation set has 5,000 images, 4,157 lesion boxes and 1,080
background boxes. `--max_train_images` samples TRAIN only, never validation.
Sample indices are saved with each run. Predictions use the original CSV schema;
metrics include mAP@0.5, mAP@0.5:0.95, lesion AP@0.5:0.95, precision, recall and F1.
`best.pt` is selected by lesion AP@0.5:0.95.

## Run

From `/home/lx/baselines/vjepa/distill/det_vjepa`:

```bash
# Full training set; defaults match the pilot optimizer/head configuration.
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_rf_detr_vitg_768.sh

# Reproduce the pilot: only the training set is sampled.
CUDA_VISIBLE_DEVICES=2,3 EPOCHS=12 EFFECTIVE_BATCH=8 \
  bash scripts/run_rf_detr_vitg_768.sh \
  --max_train_images 512 --val_every 2 --evaluate_initial --workers 2
```

Default per-GPU batch is 4, effective batch is 8, and epochs is 12. The script
derives accumulation from GPU count and rejects invalid divisibility. To use four
GPUs set both `CUDA_VISIBLE_DEVICES=0,1,2,3` and `EFFECTIVE_BATCH=16` or another
divisible value; that changes the pilot's optimization configuration. Logs and
artifacts are under `output/rf_detr_vitg_768_<timestamp>`. Existing experiment
directories cannot be overwritten.

Optimizer: AdamW, backbone LR `1e-5`, projector/head LR `1e-4`, backbone layer decay
`0.85`, weight decay `0.05`, gradient clipping `0.1`, warmup `0.5` epoch, cosine LR
with minimum ratio `0.01`, drop path `0.1`, flip probability `0.5`, seed `0`.
BF16 forward and FP32 matching/loss are used. No extra RF-DETR package, Lightning,
Transformers or pretrained model download is required in the existing environment.

Independent checkpoint evaluation, from the repository root:

```bash
cd /home/lx/baselines/vjepa
CUDA_VISIBLE_DEVICES=2 python -m distill.det_vjepa.rf_detr.infer \
  --checkpoint /absolute/path/to/best.pt --batch 8
```

This strictly reloads the RF checkpoint and runs the same full validation set,
writing a new `infer_rf_detr` directory alongside the checkpoint. It refuses to
overwrite that directory.

To start a new training schedule from an existing full detector's best weights:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH=4 EFFECTIVE_BATCH=16 EPOCHS=12 \
  bash scripts/run_rf_detr_vitg_768.sh \
  --init_weights /home/lx/baselines/vjepa/distill/det_vjepa/output/rf_detr_vitg_768_20261010T140815_2408768/best.pt \
  --lr_backbone 2e-6 --lr_head 2e-5 --evaluate_initial
```

`--init_weights` strictly loads all backbone, projector and detector weights,
then starts a fresh AdamW optimizer, epoch count, update count and warmup/cosine
schedule. `EPOCHS=12` therefore means 12 new epochs. The lower learning rates above
are a new fine-tuning configuration. Initial full validation records the starting
model's metrics before any updates. Best selection starts afresh for the new run.
`--resume` instead restores optimizer/training progress; the two options are
mutually exclusive. Both write a new experiment when used through this script.

## Verification

```bash
cd /home/lx/baselines/vjepa
OMP_NUM_THREADS=2 python -m unittest distill.det_vjepa.rf_detr.tests -v
```

Integration tests cover coordinate conversion, rejection of non-lesion targets,
empty targets, grouped train/eval query counts, all trainable parameter gradients,
the unchanged evaluator interface, and loss reduction on a fixed synthetic
optimization problem. These tests substitute a small encoder to run on CPU.

The real ViT-G hardware smoke run is in
`output/rf_detr_vitg_smoke_20261010`: two GPUs, 768 pixels, 32 sampled training
images, batch 1 per GPU, three complete optimizer updates. It completed without
nonfinite losses, missing gradients, DDP errors or OOM; peak allocated memory was
about 19,966 MiB per rank 0. This is a plumbing check, not accuracy evidence.

The real convergence pilot is in `output/rf_detr_vitg_pilot_20261010`, with 512
sampled training images, two GPUs, batch 4 per GPU, 12 epochs, initial full
validation and full validation every two epochs. Its metrics and current status
are in `metrics.jsonl`, `best_prediction.metrics.json` and `run_result.json`
(the latter appears only on successful completion). Pilot findings are recorded
separately in `pilot_report.json` once measurements are available.

At the first four completed epochs (2026-10-10 UTC), the real pilot produced:

| Completed epochs | Mean train loss | Lesion AP50 (%) | Lesion AP50:95 (%) |
|---:|---:|---:|---:|
| 0 | — | 0.00 | 0.00 |
| 1 | 13.7765 | — | — |
| 2 | 8.7957 | 65.33 | 23.68 |
| 3 | 7.5245 | — | — |
| 4 | 6.7922 | 81.74 | 43.77 |

Both post-training validation exports passed independent metric recomputation.
The epoch-2 checkpoint passed a strict model reload with all keys matching.
These measurements verify learning on the sampled training set; they do not
establish the final accuracy of a full-training-set run.

The pilot subsequently completed all 12 epochs and 768 optimizer updates.
Post-training lesion AP50:95 increased at all six validation points:
23.68%, 43.77%, 50.44%, 52.58%, 54.38%, 55.08% at epochs 2/4/6/8/10/12.
The best checkpoint is epoch 12 (zero-based checkpoint epoch 11): mAP50 85.01%,
mAP50:95 68.52%, lesion AP50 88.07%, lesion AP50:95 55.08%, precision 88.86%,
recall 88.35%, F1 88.61%. Final mean training loss is 4.6877; individual epoch
losses and AP50 values fluctuate. Peak rank-0 allocated memory is 20,512 MiB.
The completed report and full measurements remain in the run artifacts.
