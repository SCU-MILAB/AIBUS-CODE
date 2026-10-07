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

## Alpha-GIoU contingency implemented (not yet run)

Expose reg_iou_alpha=1.0 as the unchanged GIoU default; alpha=3 uses 1-IoU**3+enclosure_penalty**3, following the separately powered terms in the primary paper documented in literature/alpha_iou_2021.md. The queue places this single-variable candidate after sampler seeds 17/41 and before revisiting old coordinate changes. No model/state/decoding/evaluation modifications. Current H173/H174/H175 retain their loaded loss source.

Verified default loss and gradients bitwise equal to prior implementation, both weighted and unweighted; analytic IoU=.9 loss and relative gradient 2.43; analytic disjoint-box enclosure penalty; perfect-box zero; finite degenerate/inverted-box gradients; positive and empty-target FCOS backward connectivity; unchanged persistent criterion state; single-variable recipe and fixed time budget. Audit: output/oct7_alpha_giou_implementation_checks.json. The training CLI restricts finite alpha >=1 to avoid fractional-power singularities at zero overlap. These checks validate implementation, not detection benefit. The next scheduled candidate remains sampler_seed=17.

## H173 completed: regression-tower updates did not improve AP

H173 changed only head_train_scope from box_projection to box_tower, with the backbone remaining trainable at the incumbent learning rate. Full 5,000-image AP50:95 was 0.8098063543438911 (delta -0.0020845503), after 7,192.0 training seconds and 3,028 microbatches. Controller recorded discard (result commit d7b4d4d); no best.pt was saved. Independent replay of epoch0_prediction.csv with the unchanged metric exactly reproduces class AP and TP/FP/FN. Audit: output/oct7_h173_ap_threshold_audit.json.

Secondary AP50 delta -0.0001800 (TP -1), AP90 delta -0.0049323 (TP -5), AP95 delta -0.0077065 (TP -24). This single continuation gives no support for widening regression-tower updates under the current budget and initialization. It does not establish that this mask is universally inferior. Keep the incumbent box_projection mask.

At 16:00:28 UTC the controller filled the released slot with H176 sampler_seed=17 (protocol f8d1c7a, PID 2118186); H174/H175 remain uninterrupted. Verified H176 differs only in sampler_seed, retaining model/augmentation seed 1, reg_l1_weight=0, reg_iou_alpha=1, full validation and 120-minute training. Current verified incumbent AP remains 0.8118909046053886.

## H174/H175 completed: shorter schedules did not improve primary AP

H174 cosine_epochs=0.5: AP50:95 0.8109193697571755, delta -0.0009715348; 7,189.8 training seconds and 3,032 microbatches. H175 cosine_epochs=0.25: AP50:95 0.8108279943466187, delta -0.0010629103; 7,191.7 seconds and 3,008 microbatches. Both completed the full 5,000-image validation and time budget; controller recorded discard (4e49fb9 / 9eeec80) and no best.pt was saved. Independent unchanged-metric replays reproduce all class AP and TP/FP/FN exactly. Audits: output/oct7_h174_ap_threshold_audit.json and output/oct7_h175_ap_threshold_audit.json.

H174 AP90 rises 0.0026042 (+15 TP90), while AP95 falls 0.0018086 despite +3 TP95; H175 AP50 rises 0.0002936 (+1 TP50), while AP90/AP95 fall 0.0019935/0.0028562. Thus individual secondary diagnostics can improve without improving primary AP; short schedules have no observed overall advantage in these continuations. Retain incumbent cosine_epochs=2.0.

The controller filled the slots with H177 sampler_seed=41 (f494c28, PID 2129264, 16:28:01 UTC) and H178 reg_iou_alpha=3 (4a8c5af, PID 2132655, 16:29:04 UTC); H176 seed17 remains uninterrupted. Each new protocol was verified to change only its named variable, retaining cosine_epochs=2, model/augmentation seed 1, reg_l1_weight=0, full validation and 120-minute training. H178 retains sampler seed 0; it does not combine the pending sampling hypothesis with Alpha-GIoU. The incumbent remains 0.8118909046053886.

## Training-only IoU quality-target contingency implemented (not yet run)

Expose ctr_target=centerness as the unchanged default; ctr_target=iou replaces only positive BCE targets of the existing centerness branch with detached predicted-box/assigned-GT IoU. Regression weighting retains geometric centerness targets, assignment is unchanged, and decoding remains cls*centerness with the frozen threshold/NMS/evaluation protocol. This is an adaptation motivated by quality-supervised detection, not a reproduction of GFL; literature/quality_focal_2020.md documents the primary paper and inference differences.

Tests pass: default full FCOS outputs and gradients bitwise equal, including centerness-weighted regression and optional losses; analytic IoU targets 1/.5/0 match BCE and logit gradients; zero box gradients from quality-target BCE; empty-target outputs unchanged and every branch connected; criterion state unchanged; next proposal changes only ctr_target with alpha=1 and fixed budget. Audit: output/oct7_quality_target_implementation_checks.json. Under incumbent box_projection scope the centerness projection stays frozen; the trainable encoder can still change its input features. Moving detached IoU labels can introduce training feedback; no benefit is claimed before validation.

Queue this candidate after the two sampler-order tests and Alpha-GIoU. Current H176/H177/H178 retain loaded source/criterion state. Reload only the controller after committing this opt-in implementation; next candidate is H179 ctr_target=iou from the then-current global incumbent.

## Quality-target interpretation: existing scores already carry localization signal

Review of output/oct7_restart_incumbent_localization.json shows incumbent confidence versus per-prediction best-GT IoU Pearson r=0.5019134 over 4,238 exported lesion predictions. Confidence bins 0.25-0.50 / 0.50-0.65 / 0.65-0.80 / >=0.80 have mean best-match IoU 0.6490 / 0.8561 / 0.9078 / 0.9400 (n=423/829/1930/1056). In the top bin 88.35% reach IoU>=0.90. Therefore do not describe confidence as wholly unrelated to localization, or assume quality supervision alone will solve the strict-IoU deficit.

These are descriptive best-match statistics, not one-to-one AP, calibration probabilities or causal evidence. Scores already have a useful positive association; the pending ctr_target=iou test asks whether direct quality labels improve that association under unchanged decoding. Regression quality remains a separate limitation, which H178 tests. Keep the current queue and evaluate both hypotheses against full primary AP before adding or combining variables.

## Existing quality-output trainability contingency (not yet run)

Add head_train_scope=box_quality_projection: keep the incumbent box output/scales trainable and additionally train only the existing centerness output convolution (2,305 parameters with the actual 256-channel head). Neck, class tower/output and regression tower remain frozen; backbone trainability is independent and retains the incumbent setting. No head architecture, checkpoint keys, forward/decoding or evaluation changes. This isolates output-projection capacity after the pending ctr_target=iou proposal; it does not automatically combine label and mask changes. Each proposal is built from the then-current incumbent with exactly one changed recipe key.

Tests on the actual FCOSHead structure pass: every old scope retains identical masks for both backbone modes; new scope adds only centerness weight/bias; optimizer includes exactly the trainable parameters; state keys and initial outputs are unchanged; positive and empty-target FCOS backward gives finite, connected gradients for every trainable parameter. Audit: output/oct7_quality_projection_implementation_checks.json.

With the current incumbent this mask proposal retains ctr_target=centerness, sampler_seed=0 and reg_iou_alpha=1. The IoU-target candidate remains next (H179 if no new incumbent), and the mask candidate follows it, before revisiting old coordinate searches. Current H176/H177/H178 retain loaded source and masks. Prior H173 regression-tower negative does not establish that training this separate output projection will help or hurt; benefit is unproven until full validation.

## Sampler-order seed17 full-validation negative; IoU-target run started

H176 sampler_seed=17 completed 2,984 microbatches in 7,188.2 training seconds (time_budget_reached=true), with all 5,000 validation images. Primary lesion_AP50_95=0.8106234669685364, delta -0.00126743763685222 versus the incumbent; discard, no best.pt. Controller result commit fdc5272. Independent replay of exported predictions through unchanged metric.py exactly matches both-class AP50/AP50:95, GT and TP/FP/FN (output/oct7_h176_ap_threshold_audit.json). AP50 rises +0.000179112 (TP +1), but AP90 falls -0.003014088 (TP -6) and AP95 falls -0.005167827 (TP -20). This particular partial-epoch permutation does not improve the primary outcome; it does not rule out all sampling orders, and H177 seed41 remains pending.

Controller automatically launched H179 ctr_target=iou at 18:10:38 UTC (protocol 5e65c95, PID 2143970) alongside uninterrupted H177/H178. Verified protocol changes only ctr_target from centerness to iou, retaining sampler_seed=0, reg_iou_alpha=1, box_projection scope, incumbent initialization, 120-minute training and full 5,000-image validation. Launch gate remains 0.8118909046053886. Startup process is live; no outcome claimed before validation.

## Seed41 and Alpha-GIoU full-validation negatives

H177 sampler_seed=41: lesion_AP50_95=0.8103387475013732 (delta -0.0015521571040153503), 7,187.4 summary training seconds, 2,976 microbatches; result commit 177afeb. AP50 delta -0.000276387 / AP90 -0.003728986 / AP95 -0.005324408; corresponding TP deltas -1/-2/-16. Both tested sampler-order alternatives (17,41) are primary negatives. This does not establish that all permutations are equivalent or that data-order coverage cannot matter; retain seed0 in the incumbent recipe.

H178 reg_iou_alpha=3: lesion_AP50_95=0.8100364044308662 (delta -0.0018545001745223777), 7,185.2 summary training seconds, 2,956 microbatches; result commit f687c99. AP50 delta -0.000706077 / AP90 -0.002066016 / AP95 -0.001909345; TP deltas -3/-2/+11. The increased TP95 does not translate to improved AP95 or primary AP. Retain alpha1; this fixed-budget test does not establish Alpha-GIoU is generally ineffective.

Both runs use all 5,000 validation images, reached the training budget with finite losses, and were discarded without best.pt. Independent replay through unchanged metric.py exactly matches both-class AP50/AP50:95, GT and TP/FP/FN: output/oct7_h177_ap_threshold_audit.json and output/oct7_h178_ap_threshold_audit.json. Concurrent compute sharing and different partial-epoch prefixes remain limitations; do not interpret these single exploratory runs as significance tests. Incumbent remains 0.8118909046053886.

H180 head_train_scope=box_quality_projection launched automatically at 18:38:11 UTC (protocol 3d84d1e, PID 2154110). Verified sole changed recipe key is head_train_scope, retaining geometric centerness labels, sampler0, alpha1 and the incumbent initialization/evaluation/budget. It is separate from ongoing H179 IoU-label training; no combined-label/mask result is claimed.
