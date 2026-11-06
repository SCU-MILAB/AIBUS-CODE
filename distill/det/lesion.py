"""Prepare lesion-only labels; infer and evaluate with normal-region completion."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from distill.det.com import configure_nms
from distill.det.infer import compute_metrics, xywh_to_xyxy
from distill.det_vjepa.dataset import foreground_box


def dataset(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    root = Path(cfg.get('path') or path.parent)
    if not root.is_absolute():
        root = path.parent / root
    return cfg, root


def images_for(root, spec):
    if isinstance(spec, list):
        return sorted(set(p for item in spec for p in images_for(root, item)))
    path = Path(spec)
    if not path.is_absolute():
        path = root / path
    if path.is_dir():
        paths = sorted(str(p.absolute()) for p in path.rglob('*')
                       if p.is_file() and not any(part.startswith('.') for part in p.relative_to(path).parts)
                       and p.suffix.lower().lstrip('.') in IMG_FORMATS)
    elif path.suffix == '.txt':
        paths = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            p = Path(line.strip())
            paths.append(str(p if p.is_absolute() else path.parent / p))
    else:
        raise FileNotFoundError(path)
    if not paths:
        raise ValueError(f'No images: {path}')
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    return paths


def label_lines(image):
    path = Path(img2label_paths([image])[0])
    # Missing annotation must not silently become a normal negative.
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    for parts in lines:
        if len(parts) != 5 or parts[0] not in ('0', '1'):
            raise ValueError(f'Expected class 0/1 detection labels: {path}: {parts}')
        coords = np.asarray(parts[1:], dtype=float)
        if not np.isfinite(coords).all() or (coords < 0).any() or (coords > 1).any() or (coords[2:] <= 0).any():
            raise ValueError(f'Invalid normalized box: {path}: {parts}')
    return lines


def prepare(args):
    cfg, root = dataset(args.data)
    if len(cfg['names']) != 2:
        raise ValueError('Source YAML must contain original two-class ground truth')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}; reuse its data.yaml or choose another output')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.lesion_prepare_', dir=output.parent))
    counts = {}
    try:
        for split in ('train', 'val', 'test'):
            if not cfg.get(split):
                continue
            images = images_for(root, cfg[split])
            (stage / 'images' / split).mkdir(parents=True)
            (stage / 'labels' / split).mkdir(parents=True)
            positives = boxes = 0
            for index, image in enumerate(images):
                lines = [' '.join(row) for row in label_lines(image) if row[0] == '0']
                stem = f'{index:07d}_{hashlib.sha256(image.encode()).hexdigest()[:12]}'
                (stage / 'images' / split / (stem + Path(image).suffix)).symlink_to(Path(image).absolute())
                (stage / 'labels' / split / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
                positives += bool(lines)
                boxes += len(lines)
            counts[split] = dict(images=len(images), lesion_images=positives,
                                 negative_images=len(images)-positives, lesion_boxes=boxes)
        result = dict(path=str(output), names={0: 'lesion'}, nc=1,
                      **{split: f'images/{split}' for split in counts})
        (stage / 'data.yaml').write_text(yaml.safe_dump(result, sort_keys=False))
        (stage / 'manifest.json').write_text(json.dumps(dict(source=str(Path(args.data).resolve()),
            source_yaml_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            protocol='class0_only_keep_negative_images', counts=counts), indent=2))
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps(dict(data=str(output / 'data.yaml'), counts=counts), indent=2))


def complete(result, raw_max, black_threshold, normal_completion=True):
    detections = []
    h, w = result.orig_img.shape[:2]
    for box in result.boxes:
        cls = int(box.cls.item())
        if cls not in (0, 1) or (normal_completion and cls != 0):
            raise ValueError('Unexpected detection class')
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
        xyxy[[0, 2]] = xyxy[[0, 2]].clip(0, w)
        xyxy[[1, 3]] = xyxy[[1, 3]].clip(0, h)
        if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
            detections.append(dict(bbox=xyxy.tolist(), confidence=float(box.conf.item()),
                                   class_id=cls, class_name='lesion' if cls == 0 else 'normal'))
    if not detections and normal_completion:
        detections.append(dict(bbox=foreground_box(result.orig_img, black_threshold).tolist(),
            confidence=float(np.clip(1 - raw_max, 0, 1)), class_id=1, class_name='normal'))
    return detections


def infer(args):
    if args.batch < 1 or args.imgsz < 32 or args.imgsz % 32 or not 0 < args.conf < 1:
        raise ValueError('Require positive batch, imgsz divisible by 32, 0 < conf < 1')
    cfg, root = dataset(args.data)
    if len(cfg['names']) != 2:
        raise ValueError('Evaluation requires ORIGINAL two-class YAML, not prepared lesion-only labels')
    images = images_for(root, cfg[args.split])
    if args.max_images:
        images = images[:args.max_images]
    gt = {}
    for image in images:
        with Image.open(image) as im:
            w, h = im.size
        gt[image] = [[int(row[0]), *xywh_to_xyxy(list(map(float, row[1:])), w, h)]
                     for row in label_lines(image)]
    single_json = getattr(args, 'output_json', None)
    output = None if single_json else Path(args.output)
    if single_json:
        single_json = Path(single_json)
        single_json.parent.mkdir(parents=True, exist_ok=True)
        if single_json.exists():
            raise FileExistsError(single_json)
    else:
        output.mkdir(parents=True, exist_ok=False)
    configure_nms(args.nms_backend)
    model = YOLO(args.model_path)
    normal_completion = getattr(args, 'normal_completion', True)
    expected_classes = 1 if normal_completion else 2
    if model.model.model[-1].nc != expected_classes:
        raise ValueError(f'Expected a {expected_classes}-class checkpoint for this inference mode')
    predictions = []
    captured = []
    def capture(module, inputs, outputs):
        tensor = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[1] != 5:
            raise ValueError('Expected decoded YOLO11 single-class output [B,5,N]')
        captured.append(tensor[:, 4, :].detach().amax(dim=1).cpu().tolist())
    # Predictor may fuse the model on first use; root hook survives fusion.
    handle = model.model.register_forward_hook(capture) if normal_completion else None
    try:
        for start in range(0, len(images), args.batch):
            chunk = images[start:start + args.batch]
            captured.clear()
            results = model.predict(chunk, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                batch=args.batch, device=args.device, verbose=False, rect=False, half=False)
            if normal_completion and (not captured or len(captured[-1]) != len(results)):
                raise RuntimeError('Raw prediction batch does not match inference results')
            maxima = captured[-1] if normal_completion else [None] * len(results)
            for image, result, raw_max in zip(chunk, results, maxima):
                entry = dict(image_path=image, detections=complete(
                    result, raw_max, args.black_threshold, normal_completion=normal_completion))
                if normal_completion:
                    entry['raw_lesion_max'] = raw_max
                predictions.append(entry)
            print(f'Inference {min(start+args.batch, len(images))}/{len(images)}', flush=True)
    finally:
        if handle is not None:
            handle.remove()
    if output is not None:
        (output / 'infer.json').write_text(json.dumps(predictions))
    thresholds = [round(.5 + .05 * i, 2) for i in range(10)]
    reports = {}
    protocols = [('complement_background_perfect_match_v2', True), ('strict_box_iou', False)] if normal_completion else [('strict_box_iou', False)]
    for protocol, perfect in protocols:
        metrics, aps, per_class = compute_metrics(predictions, gt, 2, thresholds, background_perfect_match=perfect)
        metrics['lesion_AP50_95'] = float(np.mean([aps[t][0] for t in thresholds]))
        report = dict(protocol=protocol, metrics=metrics, per_class=per_class,
                      AP={str(t): classes for t, classes in aps.items()})
        reports[protocol] = report
        if output is not None:
            (output / f'{protocol}.json').write_text(json.dumps(report, indent=2))
            with (output / f'{protocol}.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['class', 'mAP@0.5 (%)', 'mAP@0.5:0.95 (%)',
                    'lesion_AP50_95 (%)', 'Precision (%)', 'Recall (%)', 'F1-score (%)'])
                writer.writeheader()
                writer.writerow(dict(zip(writer.fieldnames, ['all', *[100*metrics[k] for k in
                    ('mAP@0.5', 'mAP@0.5:0.95', 'lesion_AP50_95', 'precision', 'recall', 'f1')]])))
    protocol_metadata = dict(model=str(Path(args.model_path).resolve()),
        original_data=str(Path(args.data).resolve()), split=args.split, imgsz=args.imgsz,
        conf=args.conf, nms_iou=args.iou, black_threshold=args.black_threshold,
        normal_completion=normal_completion,
        fallback='only if zero lesion detections after confidence/NMS/clipping; score=1-raw_max' if normal_completion else None,
        ground_truth='unchanged original two-class annotations', count=len(images))
    if single_json:
        result = dict(protocol=protocol_metadata, reports=reports, predictions=predictions)
        temporary = single_json.with_suffix('.tmp')
        temporary.write_text(json.dumps(result))
        temporary.replace(single_json)
    else:
        (output / 'protocol.json').write_text(json.dumps(protocol_metadata, indent=2))
    print(json.dumps({key: value['metrics'] for key, value in reports.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--data', required=True)
    prep.add_argument('--output', required=True)
    run = sub.add_parser('infer')
    run.add_argument('--model_path', required=True)
    run.add_argument('--data', required=True, help='Original TWO-class YAML for evaluation')
    destination = run.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output', help='Existing multi-file report directory mode')
    destination.add_argument('--output_json', help='Save predictions, both metric protocols and metadata in ONE JSON')
    run.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    run.add_argument('--imgsz', type=int, default=768)
    run.add_argument('--batch', type=int, default=16)
    run.add_argument('--conf', type=float, default=.25)
    run.add_argument('--iou', type=float, default=.7)
    run.add_argument('--device', default='0')
    run.add_argument('--normal_completion', action=argparse.BooleanOptionalAction, default=True,
                     help='Disable for standard two-class YOLO inference (no normal box completion)')
    run.add_argument('--black_threshold', type=int, default=3)
    run.add_argument('--max_images', type=int, default=0)
    run.add_argument('--nms_backend', choices=['torchvision', 'torch'], default='torchvision')
    args = p.parse_args()
    (prepare if args.command == 'prepare' else infer)(args)


if __name__ == '__main__':
    main()
