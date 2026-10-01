# 2026-09-30 experiment protocol

## Fixed evaluation

- Data: `/home/lx/dataset/6cls_detection/data-2cls-remap.yaml`
- Validation: all 5,000 images, `--lesion_only --black_threshold 3`, confidence 0.25, NMS IoU 0.7
- Selection: maximize `lesion_AP50_95` as implemented by `metric.py` and `train.py`
- Reference checkpoint: `output/lesion_resolution_640_vs_768_20260928T083805_1134/768/best.pt`; lesion AP50:95 = 0.8104124888777733
- Four GPUs; at most 120 minutes of training, then full validation
- Save a checkpoint only if full validation exceeds 0.8104124888777733

## H1: shorter warmup for a time-limited continuation

Starting from the reference checkpoint, train at 768 pixels with an effective batch size of 64 (four GPUs, `--batch 4 --accum 4`). Use 0.1 warmup epochs instead of 0.5. The baseline took about 147 minutes per epoch; with a 120-minute cap, 0.5 epochs of warmup consumes most of the run. A shorter warmup should allow more useful updates at the target learning rates. This is an exploratory continuation because the reference result came from a longer six-epoch run. Correction on 2026-10-01: the original protocol incorrectly described 64 as matching the reference; the reference used two GPUs and effective batch 32. H1 changed both warmup and effective batch relative to that reference, so it does not isolate the warmup effect. H2/H3/H4 comparisons to their specified H1/H3 controls retain the same effective batch 64.

The training script change that enforces the time and checkpoint budget is run infrastructure, not the H1 model variable. If H1 fails, retain its score in `results.tsv` and return the model parameters to the baseline before the next hypothesis.

### H1 result

- 4,024 training batches; 7,192.4 training seconds; all 5,000 validation images.
- Lesion AP50:95 = 0.8066371396, below the reference by 0.0037753493. Peak allocated VRAM = 24,189.6 MiB per GPU.
- No checkpoint was written. This exploratory continuation recipe did not improve the detector within this budget; warmup and effective batch changed together relative to the reference, so no isolated warmup conclusion is supported. No H1-specific model or loss code was changed; subsequent runs still initialize from the reference checkpoint.
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

## H4: shorten the head-only cosine schedule

Keep H3's 768-pixel input, frozen backbone, optimizer, loss, initialization checkpoint, batch and evaluation fixed. Change only `--epochs` from 6 to 2, which shortens the cosine learning-rate schedule while `--train_minutes 120` still enforces the same training-time budget. H3 fell from 0.8078456610 after one epoch to 0.8048499882 after the next 3,528 batches; an earlier learning-rate reduction may preserve localization quality or improve it. Compare the full-validation run-best score to H3 and to the global reference 0.8104124889. Save weights only above the global reference.

### H4 restart after GPU recovery, 2026-10-01

The GPU wait process exited during the device recovery; H4 had not started and no training log or run directory existed. With fewer concurrent jobs, two epochs might finish before the required 120-minute budget. Before starting, separate the cosine horizon from the epoch limit: `--cosine_epochs 2 --epochs 100 --train_minutes 120`. The schedule reaches its minimum after two epochs and stays there until the training budget expires. This preserves the preregistered two-epoch decay hypothesis while avoiding an early end caused by improved throughput. Ordinary runs retain their existing schedule when `--cosine_epochs` is zero.

### H4 result, 2026-10-01

- Completed 28,031 training batches: three full epochs of 7,253 batches and a final partial epoch of 6,272 batches. The budget stop flag was true; actual training was 7,185.2 seconds, total run time 8,042.0 seconds, and peak allocated VRAM 5,586.1 MiB per GPU.
- All four validations covered 5,000 images. Lesion AP50:95 progressed through 0.8086107850, 0.8087929457, 0.8092858478 and 0.8094387755. The final epoch was also the run best.
- The run best exceeds H3 by 0.0015931144 but remains 0.0009737134 below the global reference. Status is `discard`; no checkpoint was written. The reference initialization remains unchanged for H5.
- Overall F1 at the best epoch is 0.9734178413. Lesion precision/recall are 0.9562867215/0.9788308877, with TP/FP/FN 4,069/186/88, compared with reference 4,078/213/79. F1 improved while AP did not pass the primary gate.
- Early cosine decay followed by more low-LR updates yielded a steadier trajectory than H3 in this run. H4 also processed more batches and experienced changing GPU contention, so the cross-run difference does not isolate a pure schedule effect. It does not establish that frozen-backbone adaptation is unable to improve AP.
- H5 launched at 2026-10-01 10:09 UTC, after result collection and its separate protocol commit `f5fc6d0`. It changes only feature routing relative to the preregistered H4 control recipe; H4's settings serve as that control, while checkpoint acceptance remains tied to the better original reference.

## User FPN result and next direction, 2026-10-01

The existing `vjepa_giant_multidepth_768best` experiment uses block outputs 13/26/39 and has a best full-validation lesion AP50:95 of 0.8021312609 at epoch 1. Its overall F1 is 0.9746221842 versus 0.9717087891 for the reference checkpoint; lesion-only F1 computed from existing per-class precision/recall is 0.9688846338 versus 0.9654356061. This supports the user's observation of improved F1, while high-IoU AP remains lower. These are descriptive comparisons: the prior FPN run also changed learning rates, batch accumulation, layer decay and drop path, so it does not isolate the neck effect.

After H4, prioritize a controlled FPN experiment from the reference checkpoint, changing one neck variable while retaining the encoder structure and forward behavior. Track the existing F1, per-class precision/recall and lesion AP50 alongside the primary lesion AP50:95. The checkpoint selection criterion remains lesion AP50:95. Confirm which FPN run the user meant before attributing their observation to this directory.

The exported predictions show a lower mean best-match IoU for the prior FPN run (0.88534 versus 0.89418), with the largest difference in the smallest lesion quartile (0.84025 versus 0.85648). The fraction of smallest lesions with a best-match IoU of at least 0.90 is 0.52358 versus 0.54957. This diagnostic uses the same 4,157 original lesion annotations and existing prediction CSVs, permits multiple GTs to match one prediction, and is not an AP or recall estimate. The detailed diagnostic is saved to `output/autoresearch_fpn_error_analysis_20261001.json`.

## H5: controlled multi-depth FPN, prioritized after user confirmation

On 2026-10-01, the user confirmed that their FPN observation came from `vjepa_giant_multidepth_768best`. Before H5 started, prioritize a direct controlled test of that feature routing: change only `--out_layers` from absent to `13 26 39` while retaining H4's frozen backbone, 768 input, batch/accumulation, learning rates, regularization, loss, two-epoch cosine horizon and fixed 120-minute budget. The existing output-selection interface is used; no ViT block, encoder implementation, patch/tubelet, RoPE or checkpoint keys are edited. All intermediate features use the existing encoder normalization and neck transforms.

Predict improved F1 from richer feature routing, and test whether conservative neck/head optimization can also exceed the best lesion AP50:95. Validate all 5,000 images and save only above the current best AP. Initialize from the original reference unless H4 establishes a better checkpoint. Compare the result with the old multi-depth experiment descriptively because its other training settings differed. H5 is `autoresearch_oct1_h5_multidepth`.

### H5 interim evidence: first complete validation

Epoch 0 processed 7,253 batches in approximately 48.0 minutes. Complete validation yielded lesion AP50:95 0.8023516774, lesion AP50 0.9743930101, overall F1 0.9725534734 and lesion TP/FP/FN 4,061/184/96. No checkpoint was saved. This is an interim observation; the run continues to its fixed 120-minute training budget and has no final keep/discard decision yet.

Relative to H4 at the same epoch and same 7,253 batches, AP is lower by 0.0062591076 and overall F1 by 0.0004757686. This comparison matches the number of training batches and optimizer updates, unlike comparing the eventual run bests under different throughput. It remains one exploratory run per configuration.

The read-only prediction diagnostic covers all 5,000 images and 4,157 lesions. Relative to the reference, H5 gains 11 best matches reaching IoU 0.5 and loses 28. Among the 4,050 lesions above that threshold in both runs, mean IoU changes by -0.0030028974. The smallest-area mutually matched subset changes by -0.0054948348, while the largest changes by +0.0010016095. The fraction of all lesions with a best match reaching IoU 0.9 is 0.646380 versus reference 0.670917; in the smallest quarter it is 0.503369 versus 0.549567. These summaries do not perform one-to-one matching or replace the official metric.

After this evidence, move the already proposed deeper routing 19/29/39 ahead of the generic LR/regularization sweep, following the batch, partial-fusion and weighted-regression controls. Add the independent-pyramid boundary `fpn_topdown_scale=0` after deeper routing, motivated by the primary ViTDet ablation in `literature/vitdet_pyramid_notes.md`. Each future proposal still changes one field relative to the current recipe, with a separate exact protocol commit before launch. These policy changes leave the active H5 training unchanged. Diagnostic: `output/autoresearch_oct1_h5_epoch0_diagnostics.json`.

### H5 interim evidence: second complete validation

Epoch 1 processed another 7,253 batches in approximately 48.0 minutes. Lesion AP50:95 increased to 0.8036441788, overall F1 to 0.9732853354; lesion TP/FP/FN were 4,062/179/95. The run continued into the remaining approximately 24 minutes at its minimum LR, without a final decision or checkpoint.

Both H4 and H5 had processed 14,506 batches at this point. H5 AP was lower by 0.0051487669, while overall F1 was lower by only 0.0001224436. Under this frozen-backbone continuation recipe, the remaining gap is chiefly in the primary high-IoU metric rather than a large F1 difference.

The reference-paired diagnostic still found a deficit: 10 newly above-threshold best matches and 26 lost matches at IoU 0.5, mean IoU difference -0.0022362056 among the 4,052 mutually above-threshold lesions, and -0.0043440269 in the smallest-area mutually matched subset. Those conditional subsets can differ between epochs. The unconditioned smallest-quarter fraction reaching best-match IoU 0.9 rose from 0.503369 to 0.506256, versus reference 0.549567. Complete-training results are still required for acceptance. Diagnostic: `output/autoresearch_oct1_h5_epoch1_diagnostics.json`.

### H5 final result and five-run reflection

H5 completed two full epochs plus 3,592 batches of epoch 2, totaling 18,098 batches and 7,185.2 actual training seconds. Total run time was 8,111.3 seconds; peak allocated VRAM was 5,587.2 MiB per GPU. All three validations covered 5,000 images, and the final budget-stop flag was true. The run-best/final AP50:95 was 0.8041011885, overall F1 0.9732903959, lesion TP/FP/FN 4,063/180/94. This remained 0.0063113004 below the global reference, so the result was `discard` and no checkpoint was written. The separate result commit was `623a9d9`.

At the final epoch, the descriptive reference-paired diagnostic gained 10 above-threshold best matches and lost 25 at IoU 0.5. The 4,053 mutually above-threshold lesions had mean IoU difference -0.0022301038; the smallest-area mutually matched subset (983 lesions) had difference -0.0041969744, while the largest had +0.0011714987. The smallest-quarter fraction reaching best-match IoU 0.9 was 0.510106 versus reference 0.549567. These diagnostics support continued attention to fine-level localization, with the matching and conditioning limits stated above. Artifact: `output/autoresearch_oct1_h5_epoch2_diagnostics.json`.

Across H1-H5, no fixed-budget continuation has exceeded the strong initial checkpoint. H4's faster decay produced the closest run-best AP; H5's multi-depth routing raised F1 relative to the original reference but reduced high-IoU AP under this frozen-backbone recipe. At matched two-epoch exposure, the routing gap was already visible. At run completion, H5 also processed fewer batches than H4 because of different GPU contention, so their complete-run difference combines routing and fixed-time exposure effects. The older user FPN retains different training settings and is not refuted as an entire method family by H5.

Proceed with the identified effective-batch correction before additional neck changes. Keep global checkpoint selection separate from secondary F1 improvement. Then test partial fusion, weighted regression, deeper routing and independent pyramid outputs, with conservative optimization proposals retained in the later queue. This direction follows the observed small-lesion/high-IoU gap and preserves the original encoder and evaluation.

## H6: match the reference effective batch of 32

The reference's stored configuration has `batch=4, accum=4`; its `train.log` and `scripts/run_lesion_resolution_ablation.sh` establish two GPU ranks. Its effective batch was therefore 32, while the four-GPU H1-H5 recipe has effective batch 64. Change only accumulation from 4 to 2 in the incumbent four-GPU recipe, retaining microbatch 4, frozen backbone, feature routing, learning rates, cosine horizon, loss, initialization and fixed training-time budget. The hypothesis is that restoring the reference effective batch and more frequent optimizer updates may improve the already strong detector. This is a batch-size test within the current continuation recipe, not a reproduction of the reference's entire original training run.

Initialize from the current validated best; evaluate all 5,000 images and save only above the global AP gate. Preserve the H5 FPN control at accumulation 4 so the two hypotheses remain separate.

Before the later fusion/loss tests, revisit `out_layers=13 26 39` when the current best checkpoint/recipe makes it an untested combination, for example after H6 accepts accumulation 2. Change only feature routing relative to that accepted recipe. If routing is already active or the same initialization/recipe was tested, skip it. The user's earlier FPN also had effective batch 32, motivating this conditional combination; it is not a full factorial test if both initial continuation recipes fail. Later experiment numbers may shift when this extra proposal actually runs.

H6 launched at 2026-10-01 12:24 UTC as `autoresearch_oct1_h6_accum`, following its exact protocol commit `5c72dc1`. Initialization and gate remained the original best checkpoint and 0.8104124888777733. Routing returned to the H4 control's final map (`out_layers=None`); accumulation alone changed 4→2 relative to that control. The schedule has 3,627 optimizer updates per full epoch and the same two-epoch cosine horizon. Epoch 0 validated at AP 0.808419786/F1 0.9727633819; epoch 1 validated at AP 0.809117651/F1 0.9734848485, with lesion TP/FP/FN 4,064/180/93. The 120-minute training budget and final validation remain in progress; the checkpoint gate has not been met.

## Queued: reduce top-down FPN fusion strength

After H6, change only the incumbent recipe's neck top-down contribution from 1.0 to 0.5: `fine = lateral + 0.5 * upsample(coarse)`. This should retain more of each fine lateral feature when coarse features are propagated, and may improve localization of small lesions at high IoU. This is an exploratory hypothesis suggested by the FPN result and localization diagnostic, not a claim that the previous run isolated the cause of its lower AP.

Use the incumbent feature routing and accumulation (retain H5/H6 settings only if they improve global AP), the same ViT-G encoder implementation, strides 8/16/32, head, loss, frozen backbone, 768 resolution, batch 4, warmup 0.1, two-epoch cosine horizon and 120-minute budget. The scalar introduces no parameters or checkpoint keys; old detector weights load strictly. Default 1.0 reproduces the existing neck exactly. Inference reconstructs this model setting from checkpoint arguments without changing evaluation or postprocessing.

Initialize from the best validated detector available when H7 starts. Compare H7's full-5,000-image lesion AP50:95 to that checkpoint. Record existing overall F1, lesion precision/recall and AP50 as secondary diagnostics. Save weights only above the current best AP50:95; F1 does not replace the primary selection criterion.

## Queued: target-centerness-weighted box regression

The local loss currently averages GIoU equally over positive locations. The [FCOS authors' loss implementation](https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/modeling/rpn/fcos/loss.py) passes target centerness into its regression loss and normalizes by the summed target weights; its [IoU/GIoU loss](https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/layers/iou_loss.py) supports a weighted sum. This suggests testing whether more emphasis on central positive locations improves the boxes associated with higher detection scores.

After H5-H7, change only `reg_centerness_weighted` from false to true in the incumbent recipe. Use `sum(target_centerness * GIoU_loss) / sum(target_centerness)` within the existing per-rank reduction. Retain the existing GIoU geometry, assignment, classification/centerness losses, DDP reduction, model, optimizer, evaluation and 120-minute budget. This borrows the weighting idea and is not a full reproduction of the authors' distributed normalization or IoU smoothing. Predict improved high-IoU lesion AP; the full 5,000-image lesion AP50:95 remains the selection metric. New loss settings introduce no detector parameter or checkpoint keys.

Future runs export the already computed complete validation predictions to `epochN_prediction.csv`, including runs below the checkpoint score gate. This provides evidence for localization/false-positive analysis without saving a model below the gate or changing any metrics. H4 continues with the code already loaded when it started.

## Paired FPN diagnostic refinement, 2026-10-01

The reusable read-only `scripts/analyze_predictions.py` checks the exact 5,000-image validation path set, parses unchanged original annotations, and reproduces the earlier reference/FPN best-match summaries within 1e-7. It reports descriptive GT/prediction best-match IoU, area quartiles, confidence bins, normalized box-edge errors, and paired changes. These are not official AP, precision/recall, one-to-one matches, or significance tests.

The old FPN's average GT best-match IoU drops from 0.894181 to 0.885339, but this includes missing detections. Among each run's GT boxes with best-match IoU at least 0.5, the means are 0.908620 and 0.906001; below-threshold counts are 79 and 109. On the same 4,039 GT boxes reaching 0.5 in both runs, FPN-minus-reference mean IoU is -0.003103; on the smallest-area mutually matched quarter subset (976 lesions), it is -0.003677. The FPN gains 9 above-threshold GT matches and loses 39. Thus both coverage and localization contribute to the descriptive gap; the unconditioned IoU difference should not be called pure localization degradation.

The reference has 1,215 lesion predictions with confidence at least 0.8, versus 907 for the old FPN. Confidence-bin IoU averages concern different sets of predictions and cannot isolate a causal scoring or localization effect. H5 will provide the controlled feature-routing comparison; its exported predictions can distinguish coverage loss from paired localization changes. The old FPN's other training settings remain confounders.

Diagnostic artifact: `output/autoresearch_fpn_diagnostics_20261001_v3.json`. To analyze a later export, run `python -m distill.det_vjepa.scripts.analyze_predictions --predictions <reference_best_prediction.csv> <candidate_epoch_prediction.csv> --output <new_path_inside_det_vjepa>`.

## H6 interim evidence and prospective bbox projection test, 2026-10-01

H6 epoch 0 completed 7,253 batches with full 5,000-image validation: lesion AP50:95 0.8084197864, overall F1 0.9727633819, lesion TP/FP/FN 4,066/188/91. It remains below the global checkpoint gate and continues to its full 120-minute budget; this is not a final keep/discard decision.

The reference-paired diagnostic gains 3 best matches reaching IoU 0.5 and loses 15. Among the same 4,063 lesions reaching 0.5 in both runs, mean IoU difference is -0.0005884806; in the smallest-area mutually matched subset it is -0.0003846881. The smallest-quarter fraction reaching best-match IoU 0.9 is 0.538980 versus reference 0.549567. These descriptive, non-one-to-one summaries suggest that retaining score/feature behavior while adapting boxes is a useful separate hypothesis. Artifact: `output/autoresearch_oct1_h6_epoch0_diagnostics.json`.

Epoch 1's reference-paired diagnostic gains 1 and loses 15 IoU-0.5 matches. The 4,063 mutually matched lesions improve in mean IoU by 0.0003062 overall; the smallest-area subset (991 lesions) changes by -0.0004634, and the largest by +0.0015003. All-lesion mean IoU is 0.8924023, with 93 below-threshold lesions; 54.67% of the smallest quartile reaches best-match IoU 0.9 versus 54.96% in the reference. These conditional, descriptive metrics accompany but do not replace the official AP. Artifact: `output/autoresearch_oct1_h6_epoch1_diagnostics.json`. The data continue to show small coverage loss while localization among shared matches is close to the reference, motivating the separately queued bbox-only adaptation.

After H6 and any newly eligible 13/26/39 revisit, test `head_train_scope=box_projection` against the incumbent recipe's default `all`, before the fusion/loss proposals. Freeze neck, classification tower/logits, regression tower and centerness output; update only the existing bbox output convolution and per-level distance scales. With the incumbent frozen backbone and unchanged architecture, this makes 9,223 parameters trainable and preserves raw classification/centerness outputs for identical evaluation inputs. Geometry-dependent NMS results may change. Backbone trainability remains an independent flag; score preservation does not hold if a later experiment unfreezes it.

This tests whether narrowly adapting box geometry avoids the confidence/coverage drift of broader continuation. It does not change the FCOS loss, score combination, decoder, anchors, encoder forward, model state keys, data, evaluation or 120-minute budget. Initialize from the verified global best and save only if full-validation lesion AP50:95 improves. The exact single-variable protocol is committed by the worker immediately before launch. Experiment numbers depend on conditional proposals.

Verification: `python -m distill.det_vjepa.scripts.verify_box_projection` uses the real neck, FCOS head, optimizer and loss with a small CPU stand-in encoder. A positive-plus-empty batch yields finite gradients and changed boxes while all frozen weights and raw classification/centerness outputs remain exactly equal. An all-empty batch keeps all trainable parameters connected with zero gradients. Old state keys load strictly; default trainability and independently unfrozen backbone behavior are preserved. Actual CLI parsing, fixed evaluation/budget flags, legacy recipe normalization and global-best recovery also passed. Resume rejects a changed training scope and directs callers to fresh `--init_weights` instead.
