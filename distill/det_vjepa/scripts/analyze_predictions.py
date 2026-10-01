"""Read-only localization diagnostics from the unchanged validation exports.

These best-match summaries are descriptive: they do not perform one-to-one
matching and must not be reported as official AP, precision, or recall.
Run as: python -m distill.det_vjepa.scripts.analyze_predictions --predictions
        <reference.csv> <candidate.csv> --output <diagnostics.json>
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from distill.det_vjepa.dataset import label_path_for, list_images, load_data_yaml, read_labels

DATA = '/home/lx/dataset/6cls_detection/data-2cls-remap.yaml'


def load_predictions(path, expected):
    images = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['image_path']
            size = (int(row['image_width']), int(row['image_height']))
            if key not in images:
                images[key] = dict(size=size, detections=[])
            assert images[key]['size'] == size, f'inconsistent image size: {key}'
            if row['class_id'] == '0':
                images[key]['detections'].append([
                    *[float(row[field]) for field in ('x1', 'y1', 'x2', 'y2')],
                    float(row['confidence']),
                ])
    assert set(images) == expected, 'prediction export must cover the exact 5,000 validation images'
    for item in images.values():
        item['detections'] = np.asarray(item['detections'], dtype=np.float32).reshape(-1, 5)
        assert np.isfinite(item['detections']).all(), 'nonfinite exported prediction'
    return images


def pairwise_iou(a, b):
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-6)


def summarize(values):
    result = {'n': len(values)}
    if len(values):
        result.update(mean_iou=float(np.mean(values)),
                      **{f'iou_at_least_{threshold:.2f}': float(np.mean(values >= threshold))
                         for threshold in (.5, .75, .9)})
        matched = values[values >= .5]
        result['n_below_0.50'] = int(np.sum(values < .5))
        if len(matched):
            result['mean_iou_given_ge_0.50'] = float(np.mean(matched))
    return result


def analyze(images, ground_truth, areas, bounds):
    gt_ious, prediction_ious, confidences, edge_errors = [], [], [], []
    for key, gt in ground_truth.items():
        detections = images[key]['detections']
        if not len(gt):
            prediction_ious.extend([0.] * len(detections))
            confidences.extend(detections[:, 4].tolist())
            continue
        if not len(detections):
            gt_ious.extend([0.] * len(gt))
            continue
        ious = pairwise_iou(gt, detections[:, :4])
        chosen = ious.argmax(axis=1)
        best = ious[np.arange(len(gt)), chosen]
        gt_ious.extend(best.tolist())
        prediction_ious.extend(ious.max(axis=0).tolist())
        confidences.extend(detections[:, 4].tolist())
        matched = best >= .5
        if matched.any():
            widths = np.maximum(gt[matched, 2:] - gt[matched, :2], 1e-6)
            normalizer = np.tile(widths, (1, 2))
            edge_errors.extend(((detections[chosen[matched], :4] - gt[matched]) / normalizer).tolist())
    gt_ious = np.asarray(gt_ious)
    prediction_ious, confidences = np.asarray(prediction_ious), np.asarray(confidences)
    assert len(gt_ious) == len(areas)
    quartiles = np.searchsorted(bounds[1:-1], areas, side='right')
    result = dict(all_gt_best_match=summarize(gt_ious),
                  area_quartiles=[summarize(gt_ious[quartiles == q]) for q in range(4)],
                  all_prediction_best_match=summarize(prediction_ious))
    result['confidence_bins'] = []
    edges = [.25, .5, .65, .8, 1.000001]
    for lo, hi in zip(edges, edges[1:]):
        selected = (confidences >= lo) & (confidences < hi)
        result['confidence_bins'].append(dict(lower_inclusive=lo, upper_exclusive=hi,
                                               **summarize(prediction_ious[selected])))
    if len(confidences) > 1 and np.std(confidences) > 0 and np.std(prediction_ious) > 0:
        result['confidence_iou_pearson'] = float(np.corrcoef(confidences, prediction_ious)[0, 1])
    if edge_errors:
        errors = np.asarray(edge_errors)
        result['matched_edge_errors'] = dict(
            n=len(errors), coordinates=['x1/gt_width', 'y1/gt_height', 'x2/gt_width', 'y2/gt_height'],
            mean_signed=errors.mean(axis=0).tolist(), mean_absolute=np.abs(errors).mean(axis=0).tolist())
    return result, gt_ious


def compare(reference, candidate, areas, bounds):
    mutual = (reference >= .5) & (candidate >= .5)
    delta = candidate - reference
    result = dict(candidate_only_ge_0_50=int(np.sum((reference < .5) & (candidate >= .5))),
                  reference_only_ge_0_50=int(np.sum((reference >= .5) & (candidate < .5))),
                  mean_iou_delta_all_gt=float(delta.mean()))
    groups = [('all', mutual)]
    quartiles = np.searchsorted(bounds[1:-1], areas, side='right')
    groups.extend((f'area_quartile_{q + 1}', mutual & (quartiles == q)) for q in range(4))
    result['mutually_ge_0_50'] = {}
    for name, selected in groups:
        values = delta[selected]
        details = dict(n=len(values))
        if len(values):
            details.update(mean_iou_delta=float(values.mean()),
                           fraction_improved=float(np.mean(values > 0)),
                           delta_quantiles_10_50_90=np.quantile(values, [.1, .5, .9]).tolist())
        result['mutually_ge_0_50'][name] = details
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    detector_root = Path(__file__).resolve().parents[1]
    assert args.output.resolve().is_relative_to(detector_root), 'write diagnostics only inside det_vjepa'
    assert not args.output.exists(), 'use a new output path'
    root, cfg, _ = load_data_yaml(DATA)
    paths = list_images(root, cfg['val'])
    assert len(paths) == len(set(paths)) == 5000
    predictions = [load_predictions(path, set(paths)) for path in args.predictions]
    ground_truth, areas = {}, []
    for key in paths:
        width, height = predictions[0][key]['size']
        assert all(run[key]['size'] == (width, height) for run in predictions)
        label_path = label_path_for(key)
        assert Path(label_path).is_file(), f'missing annotation: {label_path}'
        boxes, labels = read_labels(label_path, width, height)
        gt = boxes[labels == 0]
        ground_truth[key] = gt
        areas.extend(((gt[:, 2:] - gt[:, :2]).prod(axis=1) / (width * height)).tolist())
    areas = np.asarray(areas)
    bounds = np.quantile(areas, [0, .25, .5, .75, 1])
    diagnostics = [analyze(run, ground_truth, areas, bounds) for run in predictions]
    result = dict(kind='descriptive best-match IoU; not official AP or one-to-one matching',
                  data=DATA, images=len(paths), lesions=len(areas),
                  area_quartile_bounds=bounds.tolist(),
                  runs={str(path): values[0] for path, values in zip(args.predictions, diagnostics)},
                  paired_reference=str(args.predictions[0]),
                  paired_comparisons={str(path): compare(diagnostics[0][1], values[1], areas, bounds)
                                      for path, values in zip(args.predictions[1:], diagnostics[1:])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {args.output}: {len(paths)} images, {len(areas)} lesions')


if __name__ == '__main__':
    main()
