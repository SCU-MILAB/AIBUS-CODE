"""Read-only training-split diagnostic for central positive assignment."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from distill.det_vjepa.dataset import build_datasets
from distill.det_vjepa.loss import FCOSLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--radii', type=float, nargs='+', default=[1.0, 1.5])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert args.output.resolve().is_relative_to(root) and not args.output.exists()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    train, _, _ = build_datasets('/home/lx/dataset/6cls_detection/data-2cls-remap.yaml',
                                 imgsz=768, flip_prob=0, lesion_only=True, black_threshold=3)
    indices = np.random.default_rng(args.seed).choice(len(train), args.samples, replace=False)
    points, strides, levels = [], [], []
    for level, stride in enumerate((8, 16, 32)):
        grid = (torch.arange(768 // stride) + .5) * stride
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        points.append(torch.stack((x.flatten(), y.flatten()), 1))
        strides.append(torch.full((x.numel(),), float(stride)))
        levels.append(torch.full((x.numel(),), level, dtype=torch.long))
    points, strides, levels = map(torch.cat, (points, strides, levels))
    losses = {r: FCOSLoss(center_radius=r) for r in args.radii}
    stats = {r: dict(positive_counts=[], centerness_sum=0., total_positive=0,
                    positive_by_level=[0, 0, 0], lesions_without_any_positive=0) for r in losses}
    lesions = negatives = 0
    for index in indices:
        item = train[int(index)]
        boxes, labels = item['boxes'], item['labels']
        lesions += len(boxes)
        negatives += int(len(boxes) == 0)
        for radius, criterion in losses.items():
            ranges = criterion.regress_ranges[levels]
            assigned, _, ctr = criterion._match(points, ranges, radius * strides, boxes, labels)
            positive = assigned > 0
            count = int(positive.sum())
            stats[radius]['positive_counts'].append(count)
            stats[radius]['total_positive'] += count
            stats[radius]['centerness_sum'] += float(ctr[positive].sum())
            for level in range(3):
                stats[radius]['positive_by_level'][level] += int((positive & (levels == level)).sum())
            # Individual-box coverage before any overlapping-box competition.
            for box, label in zip(boxes, labels):
                assigned, _, _ = criterion._match(points, ranges, radius * strides,
                                                  box[None], label[None])
                stats[radius]['lesions_without_any_positive'] += int(not (assigned > 0).any())
    for values in stats.values():
        counts = values.pop('positive_counts')
        values['positive_count_quantiles_0_25_50_75_100'] = np.quantile(counts, [0, .25, .5, .75, 1]).tolist()
        values['mean_target_centerness'] = values.pop('centerness_sum') / max(values['total_positive'], 1)
    result = dict(kind='training_assignment_diagnostic_not_evaluation', samples=args.samples,
                  seed=args.seed, training_images=len(train), negative_images=negatives,
                  lesions=lesions, indices=indices.tolist(), resolution=768, flip_prob=0,
                  image_list_sha256=hashlib.sha256('\n'.join(train.image_paths).encode()).hexdigest(),
                  radius_results=stats)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'indices'}, indent=2))


if __name__ == '__main__':
    main()
