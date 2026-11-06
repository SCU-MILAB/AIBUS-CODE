"""Independent ViT-G + RF-DETR trainer using the unchanged project evaluator."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from ..dataset import build_datasets, collate_fn
from ..distributed import ValidationSampler, create_rank_zero_group, wait_for_rank_zero
from ..evaluation import evaluate_detector, save_predictions
from ..train import setup_distributed, lr_scale
from .model import VJepaRFDETR, RFDETRCriterion

DET = Path(__file__).resolve().parents[1]
ENCODER = DET.parents[1] / 'ckpts/vjepa_full/best_vjepa_model9639(paper).pt'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='/home/lx/dataset/6cls_detection/data-2cls-remap.yaml')
    parser.add_argument('--encoder_checkpoint', default=str(ENCODER))
    parser.add_argument('--output_dir', default=str(DET / 'output'))
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--val_batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--layer_decay', type=float, default=.85)
    parser.add_argument('--weight_decay', type=float, default=.05)
    parser.add_argument('--warmup_epochs', type=float, default=.5)
    parser.add_argument('--min_lr_ratio', type=float, default=.01)
    parser.add_argument('--clip_grad', type=float, default=.1)
    parser.add_argument('--drop_path', type=float, default=.1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--decoder_layers', type=int, default=3)
    parser.add_argument('--num_queries', type=int, default=300)
    parser.add_argument('--group_detr', type=int, default=13)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--grad_checkpoint', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--flip_prob', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sampler_seed', type=int, default=0)
    parser.add_argument('--max_train_images', type=int, default=0,
                        help='sample TRAIN only, preserving the full original validation set')
    parser.add_argument('--max_steps', type=int, default=0,
                        help='optimizer-step cap for implementation smoke tests only')
    parser.add_argument('--skip_validation', action='store_true', help='smoke test only')
    parser.add_argument('--evaluate_initial', action='store_true')
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--conf_thres', type=float, default=.25)
    parser.add_argument('--iou_thres', type=float, default=.7)
    parser.add_argument('--black_threshold', type=int, default=3)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--resume', default='',
                                help='restore model, optimizer and training progress')
    initialization.add_argument('--init_weights', default='',
                                help='load a full RF-DETR model; reset optimizer and epoch/LR progress')
    parser.add_argument('--device', default='0')
    return parser.parse_args()


def build_optimizer(model, args):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('detector.backbone.encoder.'):
            match = re.search(r'blocks\.(\d+)\.', name)
            depth = 39 - int(match.group(1)) if match else 40
            rate = args.lr_backbone * args.layer_decay ** depth
        else:
            rate = args.lr_head
        decay = args.weight_decay if parameter.ndim > 1 else 0.
        group = groups.setdefault((rate, decay), dict(params=[], base_lr=rate, lr=rate,
                                                     weight_decay=decay))
        group['params'].append(parameter)
    # Avoid the extra full-size foreach temporary for the 1B-parameter backbone.
    return torch.optim.AdamW(list(groups.values()), betas=(.9, .999), foreach=False)


def main():
    args = parse_args()
    if args.imgsz != 768:
        raise ValueError('this comparison fixes --imgsz 768')
    if min(args.batch, args.accum, args.epochs, args.val_batch) < 1:
        raise ValueError('batch, accumulation and epoch counts must be positive')
    if min(args.val_every, args.log_every) < 1:
        raise ValueError('validation and logging intervals must be positive')
    if min(args.max_train_images, args.max_steps, args.workers) < 0:
        raise ValueError('sample/step limits and worker count must be nonnegative')
    if args.conf_thres != .25 or args.iou_thres != .7 or args.black_threshold != 3:
        raise ValueError('preserve the existing confidence/NMS/foreground protocol')
    if not 0 <= args.flip_prob <= 1:
        raise ValueError('flip probability must lie in [0,1]')
    rank, world, device, distributed = setup_distributed(args.device)
    group = create_rank_zero_group(86400) if distributed else None
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_set, val_set, names = build_datasets(args.data, imgsz=768, lesion_only=True,
                                               black_threshold=3, flip_prob=args.flip_prob)
    selected = None
    if args.max_train_images:
        if args.max_train_images > len(train_set):
            raise ValueError('training sample exceeds original train set')
        selected = sorted(np.random.default_rng(args.seed).choice(
            len(train_set), args.max_train_images, replace=False).tolist())
        train_set = Subset(train_set, selected)
    train_sampler = DistributedSampler(train_set, seed=args.sampler_seed) if distributed else None
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=train_sampler,
                              shuffle=train_sampler is None, collate_fn=collate_fn,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.val_batch, shuffle=False,
                            sampler=ValidationSampler(val_set, rank, world) if distributed else None,
                            collate_fn=collate_fn, num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
    if not len(train_loader):
        raise ValueError('training subset is too small for batch/world size')
    run = Path(args.output_dir) / args.exp_name
    if rank == 0:
        if run.exists() and not args.resume:
            raise FileExistsError(f'refusing to overwrite {run}')
        run.mkdir(parents=True, exist_ok=bool(args.resume))
        cfg = dict(vars(args), lesion_only=True, model_num_classes=1,
                   selection_metric='lesion_AP50_95', world_size=world,
                   effective_batch=world * args.batch * args.accum,
                   validation_images=len(val_set), upstream_tag='1.9.4',
                   upstream_commit='9b009fa928d6218320439803d1da01869a85c072',
                   score_rule='RF-DETR lesion sigmoid; existing evaluator/NMS unchanged')
        (run / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        if selected is not None:
            (run / 'train_sample_indices.json').write_text(json.dumps(selected) + '\n')
        protected = {name: hashlib.sha256((DET / name).read_bytes()).hexdigest()
                     for name in ['evaluation.py', 'metric.py', 'dataset.py', 'distributed.py']}
        (run / 'protected_files_sha256.json').write_text(json.dumps(protected, indent=2) + '\n')
        print(f'train images {len(train_set)}, val images {len(val_set)}, effective batch {cfg["effective_batch"]}', flush=True)
    wait_for_rank_zero(group)
    model_args = {key: getattr(args, key) for key in ['hidden_dim', 'decoder_layers', 'num_queries',
                  'group_detr', 'drop_path', 'grad_checkpoint', 'freeze_backbone']}
    base = VJepaRFDETR(checkpoint=None if (args.resume or args.init_weights) else args.encoder_checkpoint,
                       **model_args).to(device)
    optimizer = build_optimizer(base, args)
    criterion = RFDETRCriterion(decoder_layers=args.decoder_layers, group_detr=args.group_detr).to(device)
    start = 0
    best = -1.
    updates = 0
    if args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location='cpu', weights_only=False, mmap=True)
        for key in ['imgsz', 'hidden_dim', 'decoder_layers', 'num_queries', 'group_detr']:
            if checkpoint['args'][key] != getattr(args, key):
                raise ValueError(f'initial weights configuration mismatch: {key}')
        base.load_state_dict(checkpoint['model'], strict=True)
        if rank == 0:
            print(f'Loaded full RF-DETR weights from {args.init_weights} '
                  f'(source epoch {checkpoint["epoch"] + 1}); '
                  'fresh optimizer, epoch 0 and optimizer step 0', flush=True)
        del checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
        for key in ['imgsz', 'hidden_dim', 'decoder_layers', 'num_queries', 'group_detr', 'freeze_backbone']:
            if checkpoint['args'][key] != getattr(args, key):
                raise ValueError(f'resume configuration mismatch: {key}')
        base.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        best = checkpoint.get('best_score', -1.)
        updates = checkpoint.get('optimizer_steps', 0)
        del checkpoint
    model = DistributedDataParallel(base, device_ids=[device.index], broadcast_buffers=False,
                                      find_unused_parameters=False) if distributed else base
    steps_per_epoch = math.ceil(len(train_loader) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = round(steps_per_epoch * args.warmup_epochs)
    if rank == 0:
        print(f'RF-DETR: {args.num_queries} queries x {args.group_detr} training groups, '
              f'{args.decoder_layers} decoder layers, {steps_per_epoch} steps/epoch', flush=True)
        if args.resume:
            print(f'Resumed full detector and optimizer; starting epoch {start + 1}, '
                  f'optimizer step {updates}', flush=True)
        elif args.init_weights:
            print('Full detector initialized from RF-DETR weights; fresh warmup/cosine schedule', flush=True)
        else:
            print('ViT-G initialized strictly from encoder checkpoint; RF-DETR projector/head start random', flush=True)

    def validate(stem, epoch):
        metrics, predictions = evaluate_detector(base, val_loader, device, len(names),
                                                  conf_thres=.25, iou_thres=.7,
                                                  lesion_only=True, group=group)
        if rank == 0:
            save_predictions(str(run / f'{stem}.csv'), predictions, names)
            result = dict(checkpoint=str(run / 'best.pt'), checkpoint_epoch=epoch,
                          class_names=names, data=args.data, imgsz=768, lesion_only=True,
                          model_num_classes=1, black_threshold=3, conf_thres=.25, iou_thres=.7,
                          selection_metric='lesion_AP50_95', **metrics)
            (run / f'{stem}.metrics.json').write_text(json.dumps(result, indent=2) + '\n')
            print(f'validation epoch {epoch}: lesion AP50 {metrics["lesion_AP50"]:.6f} '
                  f'AP50-95 {metrics["lesion_AP50_95"]:.6f}', flush=True)
        return metrics, predictions

    if args.evaluate_initial and start == 0 and not args.skip_validation:
        validate('initial_prediction', -1)
        wait_for_rank_zero(group)
    # Architectural initialization consumes different RNG counts; fix training RNG explicitly.
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        running = {}
        began = time.monotonic()
        batches = 0
        for index, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            window = min(args.accum, len(train_loader) - index // args.accum * args.accum)
            step = (index + 1) % args.accum == 0 or index + 1 == len(train_loader)
            with model.no_sync() if distributed and not step else nullcontext():
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                    outputs = model(images)
                # Official matching and losses use fp32 even when the model runs bf16.
                losses = criterion(outputs, targets)
                if not torch.isfinite(losses['loss']):
                    raise FloatingPointError(f'nonfinite loss at epoch {epoch}, batch {index}')
                (losses['loss'] / window).backward()
            batches += 1
            for name, value in losses.items():
                running[name] = running.get(name, 0.) + float(value.detach())
            if step:
                torch.nn.utils.clip_grad_norm_(base.parameters(), args.clip_grad, error_if_nonfinite=True)
                scale = lr_scale(updates, total_steps, warmup, args.min_lr_ratio)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['base_lr'] * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            if rank == 0 and (index + 1) % args.log_every == 0:
                print(f'epoch {epoch} [{index+1}/{len(train_loader)}] loss {running["loss"]/batches:.4f} '
                      f'cls {running["loss_ce"]/batches:.4f} bbox {running["loss_bbox"]/batches:.4f} '
                      f'giou {running["loss_giou"]/batches:.4f}', flush=True)
            if args.max_steps and updates >= args.max_steps:
                break
        record = dict(epoch=epoch, train={k: v / batches for k, v in running.items()},
                      train_batches=batches, optimizer_steps=updates,
                      minutes=(time.monotonic() - began) / 60)
        if not args.skip_validation and ((epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1):
            metrics, predictions = validate(f'epoch{epoch}_prediction', epoch)
            record['val_partial'] = metrics
            score = metrics['lesion_AP50_95']
            if score > best:
                best = score
                if rank == 0:
                    checkpoint = dict(model=base.state_dict(), optimizer=optimizer.state_dict(),
                                      epoch=epoch, args=vars(args), class_names=names,
                                      best_score=best, optimizer_steps=updates)
                    temporary = run / 'best.tmp.pt'
                    torch.save(checkpoint, temporary)
                    temporary.replace(run / 'best.pt')
                    save_predictions(str(run / 'best_prediction.csv'), predictions, names)
                    result = json.loads((run / f'epoch{epoch}_prediction.metrics.json').read_text())
                    result.update(selection_score=best, selection_mAP50_95=metrics['mAP50_95'])
                    (run / 'best_prediction.metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        if rank == 0:
            with (run / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(f'epoch {epoch} done: {record["minutes"]:.2f} min, loss {record["train"]["loss"]:.4f}, '
                  f'best lesion_AP50_95 {best:.6f}', flush=True)
        wait_for_rank_zero(group)
        if args.max_steps and updates >= args.max_steps:
            break
    if rank == 0:
        result = dict(status='smoke_complete' if args.max_steps else 'completed',
                      epochs_completed=epoch + 1, optimizer_steps=updates, best_lesion_AP50_95=best,
                      peak_vram_mb=torch.cuda.max_memory_allocated(device) / 2**20)
        (run / 'run_result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    wait_for_rank_zero(group)
    if distributed:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
