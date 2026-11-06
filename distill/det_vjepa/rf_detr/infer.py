"""Independent RF-DETR checkpoint inference with the unchanged evaluator/CSV schema."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..dataset import build_datasets, collate_fn
from ..evaluation import evaluate_detector, save_predictions
from .model import VJepaRFDETR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output_subdir', default='infer_rf_detr')
    args = parser.parse_args()
    path = Path(args.checkpoint).resolve()
    destination = path.parent / args.output_subdir
    destination.mkdir(exist_ok=False)
    device = torch.device('cuda', args.device)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    cfg = checkpoint['args']
    kwargs = {key: cfg[key] for key in ['hidden_dim', 'decoder_layers', 'num_queries', 'group_detr',
                                      'drop_path', 'freeze_backbone']}
    model = VJepaRFDETR(checkpoint=None, grad_checkpoint=False, **kwargs).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    _, validation, names = build_datasets(cfg['data'], imgsz=768, lesion_only=True, black_threshold=3)
    if checkpoint['class_names'] != names:
        raise ValueError('checkpoint/dataset class names differ')
    loader = DataLoader(validation, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                         collate_fn=collate_fn, pin_memory=True)
    metrics, predictions = evaluate_detector(model, loader, device, len(names),
                                               conf_thres=.25, iou_thres=.7, lesion_only=True)
    save_predictions(str(destination / 'best_prediction.csv'), predictions, names)
    result = dict(checkpoint=str(path), checkpoint_epoch=checkpoint['epoch'], data=cfg['data'],
                  imgsz=768, lesion_only=True, model_num_classes=1, class_names=names,
                  black_threshold=3, conf_thres=.25, iou_thres=.7, **metrics)
    (destination / 'best_prediction.metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
