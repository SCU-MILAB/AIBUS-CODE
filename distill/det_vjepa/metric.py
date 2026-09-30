"""
Detection metrics: mAP@0.5, mAP@0.5:0.95 and precision/recall/F1.

NMS and confidence thresholds default to the protocol reported in
distill/det/output/paper.md (conf 0.25, NMS IoU 0.7).  AP uses all-point
interpolation, matching the custom paper-baseline inference evaluator.

The shared evaluation module restores original-image coordinates and clips
boxes after NMS. This module also accepts letterbox coordinates for legacy audits.
"""

import os

import numpy as np
import torch
from torchvision.ops import nms


def postprocess(boxes, scores, conf_thres=0.25, iou_thres=0.7, max_det=300,
                pre_nms_topk=300):
    """
    boxes:  [B, N, 4] xyxy pixels
    scores: [B, N, C] class scores
    Returns a list of [m, 6] float32 arrays: x1, y1, x2, y2, score, class.
    """
    # Some CUDA/torchvision combinations cannot load the CUDA NMS PTX even
    # though model forward/backward works.  This opt-in path keeps training on
    # GPU and moves only the at-most ``pre_nms_topk`` candidates per class to
    # CPU for the non-differentiable validation/inference NMS operation.
    # Enable with DETECTION_NMS_CPU=1; this does not change the NMS algorithm
    # or its thresholds.
    nms_on_cpu = os.environ.get("DETECTION_NMS_CPU", "0") == "1"

    results = []
    num_classes = scores.shape[-1]
    for b in range(boxes.shape[0]):
        per_class = []
        for c in range(num_classes):
            cls_scores = scores[b, :, c]
            keep = cls_scores > conf_thres
            if not keep.any():
                continue
            idx = keep.nonzero(as_tuple=True)[0]
            cls_scores, cls_boxes = cls_scores[idx], boxes[b][idx]
            if cls_scores.numel() > pre_nms_topk:
                top = cls_scores.topk(pre_nms_topk).indices
                cls_scores, cls_boxes = cls_scores[top], cls_boxes[top]
            if nms_on_cpu and cls_boxes.is_cuda:
                keep_idx = nms(
                    cls_boxes.detach().cpu(),
                    cls_scores.detach().cpu(),
                    iou_thres,
                ).to(cls_boxes.device)
            else:
                keep_idx = nms(cls_boxes, cls_scores, iou_thres)
            per_class.append(torch.cat([
                cls_boxes[keep_idx],
                cls_scores[keep_idx, None],
                torch.full((keep_idx.numel(), 1), float(c),
                           device=boxes.device, dtype=boxes.dtype),
            ], dim=1))

        if per_class:
            det = torch.cat(per_class, dim=0)
            if det.shape[0] > max_det:
                det = det[det[:, 4].argsort(descending=True)[:max_det]]
            results.append(det.cpu().numpy().astype(np.float32))
        else:
            results.append(np.zeros((0, 6), np.float32))
    return results


def box_iou_np(a, b):
    """a: [n,4], b: [m,4] xyxy → [n,m] IoU."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def _nanmean(values):
    """Mean over the non-nan entries; nan if there are none."""
    values = np.asarray(values, dtype=np.float64)
    mask = ~np.isnan(values)
    return float(values[mask].mean()) if mask.any() else float('nan')


def _average_precision(scores, tp, n_gt):
    """All-point interpolated AP for one class at one IoU threshold."""
    if n_gt == 0 or len(scores) == 0:
        return float('nan')
    order = np.argsort(-scores, kind='stable')
    tp = np.asarray(tp, bool)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1, None)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(preds, gts, num_classes, iou_thresholds=np.arange(0.5, 1.0, 0.05),
             background_perfect_match=False):
    """
    preds: list of [m, 6] arrays (xyxy, score, class) from postprocess()
    gts:   list of dicts with 'boxes' [k,4] xyxy and 'labels' [k]
    Returns a dict of overall and per-class metrics.
    """
    iou_thresholds = np.asarray(iou_thresholds, np.float32)
    n_t = len(iou_thresholds)

    n_gt = np.zeros(num_classes, np.int64)
    for g in gts:
        for label in g['labels']:
            if 0 <= label < num_classes:
                n_gt[label] += 1

    # Per class / per threshold: matched flags, aligned with the scores list.
    scores_by_class = [[] for _ in range(num_classes)]
    tp_by_class = [[[] for _ in range(n_t)] for _ in range(num_classes)]

    for det, g in zip(preds, gts):
        for c in range(num_classes):
            selected = det[:, 5] == c
            if not selected.any():
                continue
            det_c = det[selected]
            order = np.argsort(-det_c[:, 4], kind='stable')
            det_boxes, det_scores = det_c[order, :4], det_c[order, 4]

            gt_boxes = g['boxes'][g['labels'] == c]
            # Optional user protocol: background is an image-level complement,
            # not an independently localized object. Wrong-background predictions
            # still have no GT to match and remain false positives.
            ious = (np.ones((len(det_boxes), len(gt_boxes)), np.float32)
                    if background_perfect_match and c == 1
                    else box_iou_np(det_boxes, gt_boxes))

            scores_by_class[c].append(det_scores)
            for ti, t in enumerate(iou_thresholds):
                used = np.zeros(gt_boxes.shape[0], bool)
                tp = np.zeros(det_boxes.shape[0], bool)
                for j in range(det_boxes.shape[0]):
                    if gt_boxes.shape[0] == 0:
                        break
                    # Highest-IoU GT that is still unmatched, like COCO matching.
                    for k in np.argsort(-ious[j]):
                        if ious[j, k] >= t and not used[k]:
                            used[k] = True
                            tp[j] = True
                            break
                tp_by_class[c][ti].append(tp)

    ap = np.full((num_classes, n_t), np.nan, np.float32)
    for c in range(num_classes):
        if not scores_by_class[c]:
            if n_gt[c] > 0:
                ap[c, :] = 0.0
            continue
        scores = np.concatenate(scores_by_class[c])
        for ti in range(n_t):
            tp = np.concatenate(tp_by_class[c][ti])
            ap[c, ti] = _average_precision(scores, tp, n_gt[c])

    # Overall precision / recall / F1 at the primary operating point (IoU 0.5).
    tp_total = fp_total = fn_total = 0
    class_tp, class_fp, class_fn = [], [], []
    for c in range(num_classes):
        if not scores_by_class[c]:
            fn_total += int(n_gt[c])
            class_tp.append(0)
            class_fp.append(0)
            class_fn.append(int(n_gt[c]))
            continue
        tp = np.concatenate(tp_by_class[c][0])
        n_pred = int(np.concatenate(scores_by_class[c]).shape[0])
        tp_total += int(tp.sum())
        fp_total += n_pred - int(tp.sum())
        fn_total += int(n_gt[c]) - int(tp.sum())
        class_tp.append(int(tp.sum()))
        class_fp.append(n_pred - int(tp.sum()))
        class_fn.append(int(n_gt[c]) - int(tp.sum()))

    class_precision = [t / max(t + f, 1) for t, f in zip(class_tp, class_fp)]
    class_recall = [t / max(t + f, 1) for t, f in zip(class_tp, class_fn)]

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        'mAP50': _nanmean(ap[:, 0]),
        'mAP50_95': _nanmean(ap),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_ap50': [_nanmean(ap[c, 0:1]) for c in range(num_classes)],
        'per_class_ap50_95': [_nanmean(ap[c]) for c in range(num_classes)],
        'num_gt': n_gt.tolist(),
        'num_pred': int(fp_total + tp_total),
        'per_class_precision': class_precision,
        'per_class_recall': class_recall,
        'per_class_tp': class_tp,
        'per_class_fp': class_fp,
        'per_class_fn': class_fn,
    }
