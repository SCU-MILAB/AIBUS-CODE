"""Shared validation: NMS in model coordinates, then original-image clipping."""
import csv
import os
import numpy as np
import torch
import torch.distributed as dist
from .distributed import ValidationSampler
from .metric import evaluate as compute_metrics, postprocess

PROTOCOL = 'original_xyxy_clip_after_nms_v1'
LESION_PROTOCOL = 'lesion_only_original_gt_complement_background_perfect_match_v2'
FIELDS = ['image_path', 'image_width', 'image_height', 'class_id', 'class_name',
          'confidence', 'x1', 'y1', 'x2', 'y2']


def restore_boxes(boxes, target):
    """Invert the exact dataset letterbox and clip to original image bounds."""
    result = np.asarray(boxes, dtype=np.float32).copy()
    ratio, pad_x, pad_y = target['letterbox']
    width, height = target['original_size']
    result[..., [0, 2]] = (result[..., [0, 2]] - pad_x) / ratio
    result[..., [1, 3]] = (result[..., [1, 3]] - pad_y) / ratio
    result[..., [0, 2]] = result[..., [0, 2]].clip(0, width)
    result[..., [1, 3]] = result[..., [1, 3]].clip(0, height)
    return result


def save_predictions(path, predictions, class_names):
    """One row per detection; an empty detection row preserves empty images."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp = path + '.tmp'
    with open(temp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for item in predictions:
            prefix = [item['image_path'], *item['original_size']]
            if not len(item['detections']):
                writer.writerow(prefix + [''] * 7)
            for det in item['detections']:
                cls = int(det[5])
                writer.writerow(prefix + [cls, class_names[cls], float(det[4]),
                                          *map(float, det[:4])])
    os.replace(temp, path)


@torch.no_grad()
def evaluate_detector(model, loader, device, num_classes, conf_thres=0.25,
                      iou_thres=0.7, max_images=0, audit=False, lesion_only=False,
                      group=None):
    """Evaluate exact global predictions; group is an all-rank Gloo group.

    Metrics are returned on every rank, predictions only on rank zero.
    Distributed loaders must use ValidationSampler (no padding/duplicates).
    """
    rank = dist.get_rank(group) if group is not None else 0
    if group is not None:
        if not isinstance(loader.sampler, ValidationSampler):
            raise ValueError('distributed evaluation requires ValidationSampler')
        loader.sampler.max_images = max_images
        max_images = 0  # The global limit is applied before sharding.
        print(f'  [rank{rank}] validation start: {len(loader.sampler)} local images',
              flush=True)
    was_training = model.training
    model.eval()
    preds, gts, predictions = [], [], []
    raw_preds, raw_gts = [], []
    clipped_count = collapsed_count = 0
    try:
        for batch_index, (images, targets) in enumerate(loader):
            boxes, scores = model.decode(model(images.to(device, non_blocking=True)))
            # Use the maximum raw lesion score before confidence filtering.
            lesion_max = scores[..., 0].float().amax(dim=1).cpu().numpy() if lesion_only else None
            dets = postprocess(boxes.float(), scores.float(), conf_thres, iou_thres)
            for image_index, (det, target) in enumerate(zip(dets, targets)):
                if max_images and len(preds) >= max_images:
                    break
                gt = target['boxes'].numpy()
                labels = target['labels'].numpy()
                if audit and not lesion_only:
                    raw_preds.append(det.copy())
                    raw_gts.append({'boxes': gt.copy(), 'labels': labels})
                restored = restore_boxes(det[:, :4], target)
                ratio, px, py = target['letterbox']
                unbounded = det[:, :4].copy()
                unbounded[:, [0, 2]] = (unbounded[:, [0, 2]] - px) / ratio
                unbounded[:, [1, 3]] = (unbounded[:, [1, 3]] - py) / ratio
                clipped_count += int(np.any(np.abs(restored - unbounded) > 1e-4, axis=1).sum())
                corrected = det.copy()
                corrected[:, :4] = restored
                keep = (restored[:, 2] > restored[:, 0]) & (restored[:, 3] > restored[:, 1])
                collapsed_count += int((~keep).sum())
                corrected = corrected[keep]
                gt_original = restore_boxes(gt, target)
                if lesion_only:
                    if num_classes != 2 or scores.shape[-1] != 1:
                        raise ValueError('lesion-only evaluation expects 1 model channel and 2 output classes')
                    # Keep every real annotation. Foreground extraction supplies
                    # a fallback prediction only; it must never create/change GT.
                    if not len(corrected):
                        confidence = float(np.clip(1.0 - lesion_max[image_index], 0, 1))
                        corrected = np.array([[*target['foreground_box'], confidence, 1]], np.float32)
                preds.append(corrected)
                gts.append({'boxes': gt_original, 'labels': labels})
                predictions.append({'image_path': target['path'],
                                    'original_size': target['original_size'],
                                    'detections': corrected})
            if (batch_index + 1) % 50 == 0:
                prefix = f'[rank{rank}] ' if group is not None else ''
                print(f'  {prefix}validation: {len(preds)} images', flush=True)
            if max_images and len(preds) >= max_images:
                break
    finally:
        model.train(was_training)
    if group is not None:
        payload = (preds, gts, predictions, raw_preds, raw_gts,
                   clipped_count, collapsed_count)
        gathered = [None] * dist.get_world_size(group) if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0, group=group)
        if rank != 0:
            result = [None]
            dist.broadcast_object_list(result, src=0, group=group)
            return result[0], []
        preds, gts, predictions, raw_preds, raw_gts = (
            [item for shard in gathered for item in shard[index]]
            for index in range(5))
        clipped_count = sum(shard[5] for shard in gathered)
        collapsed_count = sum(shard[6] for shard in gathered)
    metrics = compute_metrics(preds, gts, num_classes,
                              background_perfect_match=lesion_only)
    metrics.update(evaluation_protocol=LESION_PROTOCOL if lesion_only else PROTOCOL, num_images=len(preds),
                   num_clipped_predictions=clipped_count,
                   num_collapsed_predictions=collapsed_count)
    if lesion_only:
        metrics.update(lesion_AP50=metrics['per_class_ap50'][0],
                       lesion_AP50_95=metrics['per_class_ap50_95'][0],
                       background_confidence='1 - max raw class_0 sigmoid*centerness score',
                       background_match_rule='IoU=1 on true background; lesion misses remain FN and background FP')
    if audit and not lesion_only:
        metrics['legacy_unclipped'] = compute_metrics(raw_preds, raw_gts, num_classes)
    if group is not None:
        dist.broadcast_object_list([metrics], src=0, group=group)
    return metrics, predictions
