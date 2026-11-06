"""Re-evaluate completed detector checkpoints and export original-coordinate CSVs."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .dataset import build_datasets, collate_fn
from .evaluation import evaluate_detector, save_predictions
from .model import VJepaDetector, load_detector_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--imgsz', type=int, default=None,
                        help='override inference resolution; defaults to checkpoint resolution')
    parser.add_argument('--output_subdir', default='',
                        help='save under this subdirectory of each checkpoint directory')
    parser.add_argument('--lesion_only', action=argparse.BooleanOptionalAction, default=None,
                        help='defaults to checkpoint mode; optionally evaluate old two-class weights as lesion-only')
    args = parser.parse_args()
    if args.imgsz is not None and (args.imgsz <= 0 or args.imgsz % 16):
        parser.error('--imgsz must be positive and divisible by 16')
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    model = None
    previous_signature = None
    for path_str in args.checkpoints:
        path = Path(path_str).resolve()
        print(f'Loading {path} on {device}', flush=True)
        ckpt = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        cfg = ckpt['args']
        checkpoint_lesion_only = bool(cfg.get('lesion_only', False))
        lesion_only = checkpoint_lesion_only if args.lesion_only is None else args.lesion_only
        if checkpoint_lesion_only and not lesion_only:
            raise ValueError('a lesion-only checkpoint has no class 1 channel for two-class inference')
        imgsz = cfg['imgsz'] if args.imgsz is None else args.imgsz
        subdir = args.output_subdir or (f'infer_{imgsz}' if args.imgsz is not None else '')
        if lesion_only and not checkpoint_lesion_only and not args.output_subdir:
            subdir = (subdir or 'infer') + '_lesion_only'
        output_dir = path.parent / subdir
        stem = 'smoke_prediction' if args.max_images else 'best_prediction'
        if subdir and any((output_dir / f'{stem}{suffix}').exists()
                          for suffix in ('.csv', '.metrics.json')):
            raise FileExistsError(f'Prediction already exists in {output_dir}; choose a new --output_subdir')
        output_dir.mkdir(parents=True, exist_ok=True)
        _, val_set, names = build_datasets(cfg['data'], imgsz=imgsz,
                                          lesion_only=lesion_only,
                                          black_threshold=cfg.get('black_threshold', 3))
        if list(ckpt['class_names']) != names:
            raise ValueError('checkpoint class names differ from dataset')
        model_classes = 1 if lesion_only else len(names)
        out_layers = cfg.get('out_layers')
        out_layers = None if out_layers is None else tuple(out_layers)
        fpn_topdown_scale = cfg.get('fpn_topdown_scale', 1.0)
        box_residual = cfg.get('box_residual', 'none')
        signature = (model_classes, imgsz, tuple(cfg['fpn_strides']),
                     cfg['fpn_dim'], cfg['head_convs'], out_layers, fpn_topdown_scale, box_residual)
        if model is None or signature != previous_signature:
            del model
            torch.cuda.empty_cache()
            model = VJepaDetector(
                num_classes=model_classes, resolution=imgsz, checkpoint=None,
                fpn_strides=tuple(cfg['fpn_strides']), fpn_dim=cfg['fpn_dim'],
                head_convs=cfg['head_convs'], drop_path=cfg['drop_path'],
                grad_checkpoint=False, out_layers=out_layers,
                fpn_topdown_scale=fpn_topdown_scale, box_residual=box_residual).to(device)
            previous_signature = signature
        load_detector_weights(model, ckpt['model'], lesion_only=lesion_only)
        epoch = ckpt['epoch']
        del ckpt
        loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn,
                            pin_memory=True)
        metrics, predictions = evaluate_detector(
            model, loader, device, len(names), cfg['conf_thres'], cfg['iou_thres'],
            max_images=args.max_images, audit=True, lesion_only=lesion_only)
        # Partial smoke tests must not replace full validation exports.
        save_predictions(str(output_dir / f'{stem}.csv'), predictions, names)
        result = dict(checkpoint=str(path), checkpoint_epoch=epoch,
                      class_names=names, data=cfg['data'], imgsz=imgsz,
                      checkpoint_imgsz=cfg['imgsz'],
                      out_layers=out_layers,
                      fpn_topdown_scale=fpn_topdown_scale,
                      box_residual=box_residual,
                      lesion_only=lesion_only, model_num_classes=model_classes,
                      black_threshold=cfg.get('black_threshold', 3),
                      conf_thres=cfg['conf_thres'], iou_thres=cfg['iou_thres'],
                      **metrics)
        (output_dir / f'{stem}.metrics.json').write_text(
            json.dumps(result, indent=2) + '\n')
        legacy = metrics.get('legacy_unclipped', {}).get('mAP50_95')
        print(f'Finished {path.parent.name}: mAP50-95={metrics["mAP50_95"]:.6f}, '
              f'legacy={legacy}, '
              f'images={metrics["num_images"]}, clipped={metrics["num_clipped_predictions"]}',
              flush=True)


if __name__ == '__main__':
    main()
