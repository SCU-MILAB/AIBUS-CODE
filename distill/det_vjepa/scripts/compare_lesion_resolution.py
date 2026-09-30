"""Compare completed 640/768 runs on lesion AP and high-IoU box coverage."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from distill.det_vjepa.dataset import label_path_for, read_labels
from distill.det_vjepa.metric import box_iou_np, evaluate


def load_run(run_dir):
    config = json.loads((run_dir / 'config.json').read_text())
    saved = json.loads((run_dir / 'best_prediction.metrics.json').read_text())
    if not config['lesion_only'] or saved['num_images'] != 5000:
        raise ValueError(f'{run_dir}: expected a full 5000-image lesion-only evaluation')
    images = {}
    with (run_dir / 'best_prediction.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            path = row['image_path']
            entry = images.setdefault(path, {
                'size': (int(row['image_width']), int(row['image_height'])),
                'detections': [],
            })
            if row['class_id'] == '0':
                entry['detections'].append([
                    *(float(row[key]) for key in ('x1', 'y1', 'x2', 'y2')),
                    float(row['confidence']), 0.0,
                ])
    if len(images) != 5000:
        raise ValueError(f'{run_dir}: found {len(images)} distinct images, expected 5000')
    return config, saved, images


def measure(images, reference_sizes):
    predictions, targets = [], []
    best_ious, short_sides_at_640 = [], []
    for path in sorted(images):
        entry = images[path]
        width, height = entry['size']
        if reference_sizes[path] != (width, height):
            raise ValueError(f'original image dimensions differ for {path}')
        boxes, labels = read_labels(label_path_for(path), width, height)
        boxes = boxes[labels == 0]
        detections = np.asarray(entry['detections'], np.float32).reshape(-1, 6)
        predictions.append(detections)
        targets.append({'boxes': boxes, 'labels': np.zeros(len(boxes), np.int64)})
        ious = box_iou_np(detections[:, :4], boxes)
        best_ious.extend(ious.max(axis=0) if len(detections) else np.zeros(len(boxes)))
        short_sides_at_640.extend(
            np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
            * 640 / max(width, height)
        )

    ap = evaluate(predictions, targets, 1)
    ap90 = evaluate(predictions, targets, 1, iou_thresholds=[0.90])['mAP50']
    ap95 = evaluate(predictions, targets, 1, iou_thresholds=[0.95])['mAP50']
    best_ious = np.asarray(best_ious)
    short_sides_at_640 = np.asarray(short_sides_at_640)
    size_groups = {}
    for low, high, label in (
        (0, 16, '<16'), (16, 32, '16-32'), (32, 64, '32-64'),
        (64, 128, '64-128'), (128, np.inf, '>=128'),
    ):
        selected = (short_sides_at_640 >= low) & (short_sides_at_640 < high)
        size_groups[label] = {
            'gt': int(selected.sum()),
            'coverage_iou90': float((best_ious[selected] >= 0.90).mean())
            if selected.any() else None,
        }
    return {
        'lesion_ap50': ap['mAP50'],
        'lesion_ap50_95': ap['mAP50_95'],
        'lesion_ap90': ap90,
        'lesion_ap95': ap95,
        'gt_count': len(best_ious),
        'coverage_iou90': float((best_ious >= 0.90).mean()),
        'size_groups_at_640_px': size_groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair_dir', type=Path)
    args = parser.parse_args()
    runs = {}
    for imgsz in (640, 768):
        config, saved, images = load_run(args.pair_dir / str(imgsz))
        if config['imgsz'] != imgsz or config['selection_metric'] != 'lesion_AP50_95':
            raise ValueError(f'{imgsz}: run configuration does not match the experiment')
        runs[imgsz] = (config, saved, images)
    config640, _, images640 = runs[640]
    config768, _, images768 = runs[768]
    if set(images640) != set(images768):
        raise ValueError('640/768 validation image sets differ')
    controlled_keys = (
        'data', 'init_weights', 'lesion_only', 'black_threshold',
        'fpn_strides', 'fpn_dim', 'head_convs', 'drop_path',
        'grad_checkpoint', 'amp', 'batch', 'accum', 'val_batch',
        'epochs', 'lr_backbone', 'lr_head', 'weight_decay',
        'layer_decay', 'warmup_epochs', 'min_lr_ratio', 'clip_grad',
        'reg_weight', 'conf_thres', 'iou_thres', 'seed',
    )
    differences = [key for key in controlled_keys if config640[key] != config768[key]]
    if differences:
        raise ValueError(f'controlled settings differ: {differences}')

    reference_sizes = {path: entry['size'] for path, entry in images640.items()}
    result = {'comparison': 'same GT and fixed 640-pixel size groups; only input resolution differs'}
    for imgsz in (640, 768):
        _, saved, images = runs[imgsz]
        result[str(imgsz)] = {
            'best_epoch': saved['checkpoint_epoch'],
            **measure(images, reference_sizes),
        }
    result['delta_768_minus_640'] = {
        key: result['768'][key] - result['640'][key]
        for key in ('lesion_ap50', 'lesion_ap50_95', 'lesion_ap90',
                    'lesion_ap95', 'coverage_iou90')
    }
    path = args.pair_dir / 'comparison.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
