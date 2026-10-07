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

## Authoritative metric replay and IoU-threshold audit

A new read-only scripts/audit_prediction_ap.py invokes the existing metric.evaluate on saved full-validation exports. The standard ten-threshold lesion AP50:95, lesion AP50, and every class TP/FP/FN exactly reproduce saved best metrics within 1e-6 for both incumbent and original reference. No evaluation files were edited.

Incumbent per-threshold AP: AP50 0.9742603, AP75 0.9248235, AP90 0.5635440, AP95 0.1521101. At IoU 0.95 only 1,277 of 4,157 lesion GTs match, compared with 4,063 at IoU 0.50. The edge-supervision hypothesis therefore targets strict geometry rather than low-IoU coverage alone. Existing reference comparison and current outputs are stored in output/oct7_reference_ap_threshold_audit.json and output/oct7_incumbent_ap_threshold_audit.json. Apply the same audit to future epoch exports; do not substitute any single-threshold diagnostic for primary AP50:95 selection.

## Literature-backed contingency after the new loss/mask experiments

Verified primary source for Alpha-IoU (He et al., NeurIPS 2021): https://arxiv.org/html/2110.13675, equations 3–4 and property 3. Local source notes: literature/alpha_iou_2021.md. Candidate: change only the power of the existing GIoU overlap/penalty terms, alpha 1 to 3, after reviewing H170 and the subsequent planned loss/mask experiments. The gradient emphasis at high IoU fits the AP90/AP95 deficit and needs no architecture or inference change. This is an untested hypothesis, not an expected gain. Do not exponentiate signed GIoU as a substitute for the paper's separate powered IoU and penalty terms. Current live training source remains untouched.

## User-authorized parallel execution update

The user explicitly requested using spare GPU memory to run multiple experiments concurrently. On 2026-10-07 the controller limit was changed from one to three four-GPU jobs. H170 was preserved without restart; only its controller process was replaced and the new controller adopts its existing PID/protocol. Expected per-GPU allocation is approximately 75 GiB for three jobs, leaving approximately 20 GiB headroom. Each run retains its independent 120-minute actual training budget and full validation. Shared compute changes throughput; H170 has mixed exclusive/shared execution and its batch count must be retained when interpreting outcomes. Next two candidates change reg_l1_weight to 1.0 and 0.5 independently from the current incumbent. All future rounds use the user-authorized parallel policy, superseding the earlier serial plan.

## Schedule-duration hypothesis from live shared-GPU throughput

Observed startup plan: 1,814 optimizer steps per epoch, 3,628 total steps for cosine_epochs=2, and 181 warmup steps. With accum=4, the two fully shared-GPU loss runs reached approximately 1,900 microbatches (475 optimizer steps) after roughly 76 minutes including startup. At similar throughput a 120-minute run would perform approximately 750 optimizer steps; this is an estimate, not a completed-run measurement.

The existing step-based lr_scale would then finish at approximately 0.93 of base LR under cosine_epochs=2. Thus the time budget may stop continuation before substantial decay. Previously testing cosine_epochs=1 still leaves a long schedule relative to these shared-GPU updates. After the current loss/mask experiments, consider changing only cosine_epochs to 0.5 or 0.25 while preserving 120-minute training, 0.1-epoch warmup and all other incumbent settings. For an illustrative 750-step run, cosine_epochs=0.5 gives approximately 0.12 of base LR at the end; 0.25 reaches the fixed min_lr_ratio=0.01. Hypothesis: earlier decay reduces late continuation drift. Do not treat this rationale as proof of AP improvement. H170 mixes exclusive and shared compute and has substantially more updates, so completed-run batch counts remain essential context.

## Completed H170: relative boundary weight 0.25 — discard

Full-validation lesion_AP50_95 = 0.8106899559497833, versus incumbent 0.8118909046053886 (delta -0.0012009487). Training = 7,186.2 seconds, 5,016 microbatches, one timed-out partial epoch; peak allocated VRAM = 24,025.2 MiB (23.5 GiB). Full validation = 5,000 images. No best.pt was saved; controller recorded discard and committed the result before launching H173.

The unchanged metric replay reproduces AP and all class TP/FP/FN. Compared with incumbent: AP50 +0.0003711; AP90 -0.0045688; AP95 -0.0060234, with 21 fewer TP at IoU 0.95. The candidate's descriptive small-area-quartile IoU >= 0.90 coverage is 54.09% versus incumbent 54.19%; mutually covered GT mean best-IoU change is -0.0001920. Thus this run does not support the predicted strict-localization improvement. Do not conclude every boundary-loss weight fails: H171 (1.0) and H172 (0.5) remain running. H170 mixed exclusive/shared GPU execution and processed more batches than historical shared runs, so this is not a clean causal isolation of the loss term.

Artifacts: output/oct7_h170_ap_threshold_audit.json, output/oct7_h170_localization_comparison.json, and output/autoresearch_oct7_h170_reg_l1_weight_0p25.result.json. Controller automatically launched H173 with head_train_scope box_projection -> box_tower, all other incumbent recipe fields unchanged; reg_l1_weight remains zero. Its protocol is experiments/autoresearch_oct7_h173_head_train_scope_box_tower/protocol.json.

## Dispatch update while H171–H173 remain live

After the negative H170 result and the measured shared-GPU update counts, prioritize cosine_epochs 0.5 and 0.25 for the next available slots. Each candidate changes only decay duration from the then-current incumbent, retaining 120-minute training and 0.1-epoch warmup. H171, H172 and H173 remain untouched. This updates the earlier ordering that deferred schedule tests until all loss/mask runs finished; independent tests can overlap in the user-authorized three-run queue. Keep backbone-rate and weight-decay alternatives available afterward. Reload only the controller so it reads the new proposal order and adopts the same live training PIDs.

## Additional training-only quality-supervision contingency

Verified GFL (NeurIPS 2020) equation 2 and inference description from the primary PDF; source notes are literature/quality_focal_2020.md. The original method changes inference scoring and regression representation, which are outside the frozen evaluation/decoding protocol. A future single-variable adaptation could replace geometric centerness BCE targets with detached IoU targets, retaining the same model keys and cls*centerness decoding. A classification QFL adaptation is another candidate but needs careful consideration of frozen classification projections. These are unimplemented, untested candidates after the short-decay tests, and carry no claim of expected improvement. H171 and H172 remain in full validation and H173 remains in training.

## Completed boundary-loss family and follow-up queue

H171 (reg_l1_weight 1.0): AP50:95 0.8098883658647538, delta -0.0020025387; 7,194.5 training seconds, 3,036 microbatches. H172 (0.5): AP50:95 0.8106036633253098, delta -0.0012872413; 7,193.4 seconds, 3,012 microbatches. Both passed 5,000-image/full-budget checks, were recorded as discard, and saved no best.pt. Unchanged-metric replays reproduce AP and all class TP/FP/FN.

The three tested boundary weights (0.25/0.5/1.0) did not improve primary AP. H172 has a small secondary AP95 increase (+0.0008072, +9 TP95), while AP50 and primary AP fall. Thus do not claim every localization diagnostic worsened; the predicted overall gain is unsupported. H171 AP95 falls 0.0053364. Detailed audits: output/oct7_h171_ap_threshold_audit.json, output/oct7_h172_ap_threshold_audit.json and output/oct7_boundary_loss_localization_comparison.json.

The controller launched H174 (cosine_epochs 0.5) and H175 (0.25), each differing only in schedule duration from the incumbent; boundary-loss weight is zero. H173 continues its regression-tower training-mask experiment. All three live runs retain sampler seed 0.

## Verified DDP sampler-order limitation and opt-in experiment

Current train.py previously constructed DistributedSampler without a seed argument. PyTorch 2.9.1's installed implementation uses a private generator seeded with sampler.seed + sampler.epoch; the default is 0. New init_weights experiments start at epoch 0. Therefore changing torch.manual_seed / --seed does not change the DDP permutation. Earlier --seed tests vary augmentation/model randomness and should not be interpreted as changes to DDP image order. Official versioned documentation: https://docs.pytorch.org/docs/2.9/data.html#torch.utils.data.distributed.DistributedSampler . Context7 tools are not available in this session; the local implementation, executable checks and official documentation were used.

Read-only reproduction: global torch seeds 0, 1 and 2 produce identical full rank-0 index sequences. A four-rank prefix of 3,000 microbatches at batch 4 covers the same 48,000 of 116,059 training images (41.36%) in repeated epoch-0 runs. With sampler seed 17, the same-size prefix contains 28,078 different image indices. This diagnoses repeated coverage in partial-epoch continuations; it does not imply the original pretrained detector never saw the remaining training images, or prove the cause of the AP plateau. Audit: output/oct7_sampler_order_audit.json.

Expose sampler_seed as an opt-in DDP permutation control, default 0 to preserve all prior/current behavior. The argument is identical across ranks; never add rank to this shared seed. Training membership/splits/labels/augmentation probability and all validation are unchanged. After the queued short-decay tests, test sampler_seed 17 and 41 as independent single-variable proposals from the then-current incumbent, with the model/augmentation seed unchanged. The current three processes retain their loaded source and sampler state. Reload only the controller to pick up future proposals after the implementation and invariants pass.
