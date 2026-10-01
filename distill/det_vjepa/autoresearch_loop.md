# Persistent detector experiment loop

The loop runs on `autoresearch/sep30` using `scripts/run_autoresearch_h4_h5.py`. It first collects the already running H4, then tests the user-confirmed multi-depth FPN under the same optimization recipe (H5), then halves the top-down fusion strength (H6).

Every experiment uses four GPUs and a 120-minute training budget. Validation uses all 5,000 images from the fixed remapped dataset with lesion-only mode, black threshold 3, confidence 0.25 and NMS IoU 0.7. The primary metric is maximized lesion AP50:95. F1, lesion precision/recall and AP50 are diagnostics. Evaluation, dataset, postprocessing and encoder implementation are unchanged.

Each new experiment has one changed recipe field. Before launch the loop commits `experiments/<exp_name>/protocol.json`, including the variable, old/new values, rationale, current best checkpoint, score gate and exact command. After a completed run it verifies the training budget, finite loss, full validation, score summary and checkpoint gate; writes five-column `results.tsv`; and commits a readable entry to `autoloop_log.md`. Invalid or failed runs receive `crash`. A successful run updates the initialization checkpoint and incumbent recipe only when it exceeds the current best AP50:95. A discarded run's CLI settings are reverted for the next proposal; its protocol and result remain as evidence.

After H6, the named single-variable tests are: target-centerness-weighted GIoU (H7), stochastic depth 0, head LR 1e-5, GIoU weight 1.5, FPN fusion 0.75, cosine horizon 1 epoch, backbone trainability, accumulation 2, GIoU weight 2, head LR 5e-6, FPN fusion 0.25, cosine horizon 3 epochs, and alternative existing feature routing 19/29/39. Equal or already tested recipes are skipped. Subsequent passes refine one numeric coordinate around the incumbent, with shrinking steps and bounded values. This is exploratory optimization; the loop does not treat repeated validation comparisons as confirmatory evidence.

Each subsequent run exports its already computed full-validation predictions as `epochN_prediction.csv`, even when AP is below the checkpoint gate. This enables analysis of failed hypotheses while retaining the same evaluation and checkpoint rule.

Resources are checked before each launch. Head-only runs require at least 16,000 MiB free per GPU; full tuning requires 32,000 MiB. CUDA must expose four devices and disk must have at least 20 GiB free. The loop waits when resources are unavailable and never interrupts another training job. Checkpoints are saved only above the current best AP gate.

Runtime logs and receipts live in the ignored `output/` directory. The loop PID is in `output/autoresearch_h4_h5.queue.pid`, its log is `output/autoresearch_h4_h5.queue.log`, and each subsequent training PID is stored in `output/<exp_name>.pid`. Creating `output/autoresearch.stop` requests an exit after the current experiment finishes; the marker is checked before the next launch. A restart can collect an existing experiment or use its saved receipt instead of launching it twice.

The persistent process executes this registered experiment policy. Its later proposals follow the bounded refinement rules above; deeper code changes and new research directions can be introduced in an active agent session.
