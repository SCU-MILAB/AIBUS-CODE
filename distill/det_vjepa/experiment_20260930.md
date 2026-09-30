# 2026-09-30 experiment protocol

## Fixed evaluation

- Data: `/home/lx/dataset/6cls_detection/data-2cls-remap.yaml`
- Validation: all 5,000 images, `--lesion_only --black_threshold 3`, confidence 0.25, NMS IoU 0.7
- Selection: maximize `lesion_AP50_95` as implemented by `metric.py` and `train.py`
- Reference checkpoint: `output/lesion_resolution_640_vs_768_20260928T083805_1134/768/best.pt`; lesion AP50:95 = 0.8104124888777733
- Four GPUs; at most 120 minutes of training, then full validation
- Save a checkpoint only if full validation exceeds 0.8104124888777733

## H1: shorter warmup for a time-limited continuation

Starting from the reference checkpoint, train at 768 pixels with the same effective batch size of 64 (`--batch 4 --accum 4`). Use 0.1 warmup epochs instead of 0.5. The baseline took about 147 minutes per epoch; with a 120-minute cap, 0.5 epochs of warmup consumes most of the run. A shorter warmup should allow more useful updates at the target learning rates. This is an exploratory continuation because the reference result came from a longer six-epoch run.

The training script change that enforces the time and checkpoint budget is run infrastructure, not the H1 model variable. If H1 fails, retain its score in `results.tsv` and return the model parameters to the baseline before the next hypothesis.

### H1 result

- 4,024 training batches; 7,192.4 training seconds; all 5,000 validation images.
- Lesion AP50:95 = 0.8066371396, below the reference by 0.0037753493. Peak allocated VRAM = 24,189.6 MiB per GPU.
- No checkpoint was written. The shorter warmup did not improve the detector within this budget. No H1-specific model or loss code was changed; subsequent runs still initialize from the reference checkpoint.
- The run shared the GPUs with one existing detector training job and, later, a second multi-depth detector job. `train_batches` documents the resulting throughput.

## Baseline error analysis for the next hypothesis

For the 5,000-image validation predictions already on disk, the smallest quarter of 4,157 annotated lesions has mean best-match IoU 0.8565 at 768 pixels, versus 0.9159 for the largest quarter. Comparing the old 640- and 768-pixel prediction files on the same objects, the smallest-quarter mean best-match IoU rises from 0.8378 to 0.8565 while the largest quarter changes from 0.9167 to 0.9159. This is descriptive error analysis, not a change to the official AP computation.

## H2: increase input resolution to 896

Keep the H1 time-limited optimization recipe, initialization checkpoint, batch size, warmup, loss, and evaluation fixed. Change only `--imgsz` from 768 to 896. The prediction analysis above suggests that the extra spatial detail may improve localization of small lesions, which have the largest gap at high IoU. Predict a higher lesion AP50:95 than H1's 0.806637 and test whether it also exceeds the global reference 0.810412. Use the same 120-minute training budget and 5,000-image validation; save a checkpoint only above the global reference.

### H2 result

- 2,520 training batches; 7,190.0 training seconds; all 5,000 validation images.
- Lesion AP50:95 = 0.8049352869, below both H1 and the global reference. Peak allocated VRAM = 25,071.1 MiB per GPU.
- No checkpoint was written. The 896-pixel run had much lower throughput than H1 while two other four-GPU jobs shared the machine, so this result measures the fixed-time recipe under that contention and does not isolate a pure resolution effect.

## H3: train the detection neck and head with a frozen backbone

Return to H1's 768-pixel recipe and the same reference checkpoint. Change only whether backbone parameters receive gradients: `--freeze_backbone` disables their updates while retaining the ViT-G structure, weight loading, and forward behavior. The hypothesis is that the pretrained features are already strong and that more neck/head optimization steps within 120 minutes will improve localization without perturbing the backbone. Compare H3 to H1's 0.806637 under the same 768-pixel input and to the global reference 0.810412. Keep the full 5,000-image validation and checkpoint gate unchanged.

### H3 result

- Epoch 0: 7,253 batches, lesion AP50:95 = 0.8078456610 on 5,000 images.
- Epoch 1 stopped after another 3,528 batches at 7,185.5 cumulative training seconds; lesion AP50:95 = 0.8048499882 on 5,000 images.
- The run-best score remained below 0.810412, so no checkpoint was written. Peak allocated VRAM = 5,586.1 MiB per GPU.
- Freezing the backbone increased throughput and reduced memory, but validation quality declined during further head-only adaptation. A faster learning-rate decay is the next single-variable test.
