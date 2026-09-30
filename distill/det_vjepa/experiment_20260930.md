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
