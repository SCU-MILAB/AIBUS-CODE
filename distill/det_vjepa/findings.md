# Detector research findings — 2026-10-07 restart

Primary objective: maximize lesion_AP50_95. The minimize wording in program.md contradicts AP semantics, existing keep/discard decisions and the checkpoint selection code, and is treated as a typo.

## Evidence and current best

Verified incumbent: output/autoresearch_oct3_h72_reg_weight_0p5/best.pt, lesion_AP50_95 = 0.8118909046053886 on 5,000 validation images. Original 768 reference = 0.8104124888777733. Incumbent validation AP50 = 0.9742602705955505: recall is already high, so stricter localization is a plausible target.

Recent local parameter refinements repeatedly degraded AP. The previous controller shared all four GPUs across up to four DDP jobs. This reduces batches processed within a fixed wall-clock training budget; do not interpret serial versus shared-GPU differences as isolated algorithm effects.

The user authorized stopping the stalled loop. Old H167, H168 and H169 were interrupted, archived as crash because the TSV schema has no interrupted status, and explicitly flagged interrupted_by_user in their receipts. They provide no completed-budget conclusions. Logs and checkpoints were retained.

## New exploratory hypotheses

1. Relative boundary Smooth L1 alongside GIoU may improve high-IoU localization. Test reg_l1_weight 0.25, 1.0, 0.5, one at a time; all other incumbent settings fixed. Normalization uses each GT width/height; Smooth L1 beta = 0.1. The auxiliary term is inside reg_weight. Default zero preserves the original objective and old checkpoint compatibility.
2. Training the regression tower as well as its output projection may release a capacity constraint. Change only head_train_scope to box_tower. Centerness shares that tower, so its outputs may also change even though the centerness projection stays frozen.
3. Broader backbone learning-rate / weight-decay interventions remain available; do not infer outcomes from interrupted old jobs.

## Execution constraints

New branch autoresearch/oct7 descends from the existing research branch to preserve source changes, receipts and untracked user files. No master branch exists; main is not used to reset the research tree. Historical results.tsv is retained.

Run only one four-GPU job at a time. Fix data, split, annotations, backbone structure/loading/forward, two-frame input, metric.py, evaluation.py, NMS thresholds (0.25 confidence, 0.7 IoU) and full 5,000-image validation. Each run trains for 120 minutes excluding startup and evaluation, using the existing training-time budget code. Save checkpoints only above the incumbent AP. Protocol is committed before execution; results are collected and committed afterward.

Sanity checks passed before launch: perfect boxes incur zero auxiliary loss; perturbed boxes incur greater loss and different boundary gradients; empty-target backward stays finite; regression tower training mask is correct. Read samples from train and validation without changing the dataset.

The controller continuously records results, updates the global incumbent, skips tested recipes and starts the next single-variable experiment. Stop marker: output/autoresearch.stop. Status: output/autoresearch_live_status.md. Agent should reassess hypotheses after the first four new runs and adjust the plan if none improves AP, rather than relying only on diminishing coordinate perturbations.

## Read-only localization audit after restart

Source: output/oct7_restart_incumbent_localization.json, generated from existing full-validation prediction CSVs and unchanged GT. These are best-match descriptive IoUs, not one-to-one evaluation and not official recall/AP.

- Incumbent IoU >= 0.90 coverage: 67.09% of all 4,157 lesions; smallest area quartile 54.19%, largest 74.06%. Relative mean absolute edge errors: x1 2.35%, y1 2.49%, x2 2.45%, y2 2.63% of GT width/height.
- Relative to the original 768 checkpoint, incumbent official AP improves by 0.0014784, but descriptive mean best IoU drops 0.0023547 and the number below IoU 0.50 rises from 79 to 94. There are 2 newly covered GTs versus 17 newly uncovered GTs at IoU 0.50. Mean IoU improvement among mutually covered GTs is only 0.0002099. Thus the existing AP gain is not evidence of broadly better geometric localization; scoring/FP tradeoffs may contribute.
- Original and incumbent aggregate IoU >= 0.90 coverage are identical, despite AP differences. Next-run diagnostics should compare the incumbent export against the candidate epoch export and inspect small-lesion coverage, misses, signed edge errors and confidence bins alongside official AP.
- H170 is exploratory and combines a new loss intervention with exclusive GPU execution relative to past shared-GPU runs. Attribute any gain cautiously; serial versus shared-GPU comparisons have different training throughput within the fixed time budget. No claim of statistical significance is justified by these single-seed adaptive continuation experiments.
