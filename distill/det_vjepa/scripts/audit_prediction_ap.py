"""Audit saved validation exports using the unchanged authoritative metric.

Single-threshold AP values are secondary diagnostics. The standard ten-threshold
AP is recomputed first and must match the saved best metrics when supplied.
No inference, NMS, dataset, or evaluation implementation is changed.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from distill.det_vjepa.dataset import label_path_for, list_images, load_data_yaml, read_labels
from distill.det_vjepa.metric import evaluate


def audit(prediction_path, expected_metrics=None):
    root, cfg, _ = load_data_yaml('/home/lx/dataset/6cls_detection/data-2cls-remap.yaml')
    paths = list_images(root, cfg['val'])
    assert len(paths) == len(set(paths)) == 5000
    images = {}
    with prediction_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['image_path']
            size = (int(row['image_width']), int(row['image_height']))
            item = images.setdefault(key, {'size': size, 'detections': []})
            assert item['size'] == size
            if row['class_id']:
                item['detections'].append([
                    *[float(row[field]) for field in ('x1', 'y1', 'x2', 'y2')],
                    float(row['confidence']), int(row['class_id'])])
    assert set(images) == set(paths), 'export must cover the exact validation images'
    predictions, ground_truth = [], []
    for path in paths:
        width, height = images[path]['size']
        assert Path(label_path_for(path)).is_file()
        boxes, labels = read_labels(label_path_for(path), width, height)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        detections = np.asarray(images[path]['detections'], np.float32).reshape(-1, 6)
        assert np.isfinite(detections).all()
        predictions.append(detections)
        ground_truth.append({'boxes': boxes, 'labels': labels})
    full = evaluate(predictions, ground_truth, 2, background_perfect_match=True)
    if expected_metrics:
        expected = json.loads(expected_metrics.read_text())
        assert expected['num_images'] == 5000
        assert abs(full['per_class_ap50_95'][0] - expected['lesion_AP50_95']) < 1e-6
        assert abs(full['per_class_ap50'][0] - expected['lesion_AP50']) < 1e-6
        assert full['per_class_tp'] == expected['per_class_tp']
        assert full['per_class_fp'] == expected['per_class_fp']
        assert full['per_class_fn'] == expected['per_class_fn']
    by_threshold = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        metrics = evaluate(predictions, ground_truth, 2,
                           iou_thresholds=[threshold], background_perfect_match=True)
        by_threshold.append({'iou': round(float(threshold), 2),
                             'lesion_ap': metrics['per_class_ap50'][0],
                             'tp': metrics['per_class_tp'][0],
                             'fp': metrics['per_class_fp'][0],
                             'fn': metrics['per_class_fn'][0]})
    return {'kind': 'unchanged metric applied to saved post-NMS export',
            'predictions': str(prediction_path), 'images': len(paths),
            'expected_metrics': str(expected_metrics) if expected_metrics else None,
            'standard_metrics': full, 'lesion_by_iou': by_threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--expected_metrics', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1])
    assert not args.output.exists(), 'use a new output path'
    result = audit(args.predictions, args.expected_metrics)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['lesion_by_iou']))


if __name__ == '__main__':
    main()
