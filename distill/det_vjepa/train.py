"""
Full-parameter detection training with a V-JEPA ViT-G backbone.

The entire network — ViT-G encoder, simple feature pyramid and FCOS head — is
trainable; nothing is frozen and no teacher is involved.  ViT-G is 1012M params,
so use gradient checkpointing (on by default), bf16 autocast and gradient
accumulation.

Single GPU:
    python -m distill.det_vjepa.train --data /path/to/data.yaml

Multi GPU (4 GPUs, batch 8 per GPU → effective 32):
    torchrun --nproc_per_node=4 -m distill.det_vjepa.train \
        --data /path/to/data.yaml --batch 8
"""

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
_VJEPA_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from distill.det_vjepa.dataset import build_datasets, collate_fn
from distill.det_vjepa.loss import FCOSLoss
from distill.det_vjepa.evaluation import evaluate_detector, save_predictions
from distill.det_vjepa.model import VJepaDetector, load_detector_weights
from distill.det_vjepa.distributed import (
    ValidationSampler, create_rank_zero_group, wait_for_rank_zero,
)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed(device_arg):
    """Returns (rank, world_size, device, is_distributed)."""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        return dist.get_rank(), world_size, torch.device('cuda', local_rank), True

    if not device_arg:
        device_arg = '0'
    if ',' in device_arg:
        raise ValueError(
            f'--device {device_arg!r} lists several GPUs; launch with '
            f'torchrun --nproc_per_node={len(device_arg.split(","))} instead'
        )
    device = torch.device(device_arg if device_arg.startswith('cuda')
                          else f'cuda:{device_arg}')
    torch.cuda.set_device(device)
    return 0, 1, device, False


# ---------------------------------------------------------------------------
# Optimizer: layer-wise LR decay over the backbone
# ---------------------------------------------------------------------------

def build_optimizer(model, args, num_blocks):
    """
    AdamW with layer-wise LR decay on the backbone.

    Following ViTDet / BEiT fine-tuning: each transformer block gets
    lr * layer_decay^(depth - 1 - block_index), so the blocks closest to the
    input move least and the pretrained features are preserved.  Weight decay is
    applied to matrices only, never to norms or biases.
    """
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            match = re.search(r'blocks\.(\d+)\.', name)
            depth = num_blocks - 1 - int(match.group(1)) if match else num_blocks
            base_lr = args.lr_backbone * args.layer_decay ** depth
        else:
            base_lr = args.lr_head

        decay = param.ndim > 1
        key = (round(base_lr, 12), decay)
        if key not in groups:
            groups[key] = {
                'params': [],
                'base_lr': base_lr,
                'weight_decay': args.weight_decay if decay else 0.0,
            }
        groups[key]['params'].append(param)

    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999))


def lr_scale(step, total_steps, warmup_steps, min_ratio):
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, base_model, loader, optimizer, criterion, device,
                    epoch, args, steps_per_epoch, total_steps, warmup_steps,
                    is_distributed, log, remaining_seconds=None):
    model.train()
    running = torch.zeros(4, device=device)  # loss, cls, reg, ctr
    started = None
    batches = 0
    timed_out = False
    accum = max(args.accum, 1)
    # `lr_scale` is defined over the complete run, not over an individual
    # epoch.  Starting at the epoch offset also makes a resumed run use the
    # appropriate point on the warmup/cosine schedule.
    global_step = epoch * steps_per_epoch

    for it, (images, targets) in enumerate(loader):
        if started is None:
            # DataLoader worker startup is outside the training budget.
            started = time.monotonic()
        batches += 1
        images = images.to(device, non_blocking=True)
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in t.items()} for t in targets]

        is_step = (it + 1) % accum == 0 or (it + 1) == len(loader)
        # DDP no_sync must cover the forward pass as well as backward.
        sync_context = model.no_sync() if is_distributed and not is_step else nullcontext()
        window_size = min(accum, len(loader) - (it // accum) * accum)
        with sync_context:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                head_out = model(images)
                losses = criterion(head_out, targets, base_model.anchor_points,
                                   base_model.anchor_strides, base_model.anchor_levels)
                loss = losses['loss'] / window_size
            loss.backward()

        running += torch.stack([
            losses['loss'].detach(), losses['cls'], losses['reg'], losses['ctr'],
        ])

        if not is_step:
            continue

        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.clip_grad)

        scale = lr_scale(global_step, total_steps, warmup_steps, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * scale
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if remaining_seconds is not None:
            # Leave room for the last optimizer step to finish inside the
            # 120-minute budget. Rank zero decides so every DDP rank exits
            # after the same step.
            stop = torch.tensor(
                int(time.monotonic() - started >= max(remaining_seconds - 15, 0))
                if not is_distributed or dist.get_rank() == 0 else 0,
                device=device, dtype=torch.int32)
            if is_distributed:
                dist.broadcast(stop, src=0)
            timed_out = bool(stop.item())

        if log and (it + 1) % args.log_every == 0:
            avg = running / (it + 1)
            print(f'  epoch {epoch} [{it + 1}/{len(loader)}] '
                  f'loss {avg[0]:.4f} (cls {avg[1]:.4f} reg {avg[2]:.4f} '
                  f'ctr {avg[3]:.4f}) lr {optimizer.param_groups[-1]["lr"]:.2e}',
                  flush=True)
        if timed_out:
            break

    if is_distributed:
        running = running.clone()
        dist.all_reduce(running)
        running /= dist.get_world_size()
    avg = (running / batches).tolist()
    return ({'loss': avg[0], 'cls': avg[1], 'reg': avg[2], 'ctr': avg[3]},
            time.monotonic() - started, timed_out, batches)


def evaluate(base_model, loader, device, num_classes, args, max_images=0,
             return_predictions=False, group=None):
    metrics, predictions = evaluate_detector(
        base_model, loader, device, num_classes, args.conf_thres,
        args.iou_thres, max_images=max_images,
        lesion_only=getattr(args, 'lesion_only', False), group=group)
    return (metrics, predictions) if return_predictions else metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='ViT-G (V-JEPA) + FCOS full-parameter detection training')

    p.add_argument('--data', type=str, required=True, help='YOLO dataset yaml')
    p.add_argument('--ckpt', type=str,
                   default=os.path.join(_VJEPA_ROOT, 'ckpts', 'vjepa_full',
                                        'best_vjepa_model9639(paper).pt'),
                   help='V-JEPA checkpoint providing the ViT-G encoder')
    p.add_argument('--output_dir', type=str, default=os.path.join(_HERE, 'output'))
    p.add_argument('--exp_name', type=str, default='vjepa_giant_fcos')

    # Model
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--fpn_strides', type=int, nargs='+', default=[8, 16, 32])
    p.add_argument('--out_layers', type=int, nargs='+', default=None,
                   help='zero-based ViT block outputs, one per FPN level; '
                        'for example 13 26 39. Default: final block for every level')
    p.add_argument('--fpn_dim', type=int, default=256)
    p.add_argument('--head_convs', type=int, default=4)
    p.add_argument('--drop_path', type=float, default=0.0)
    p.add_argument('--lesion_only', action=argparse.BooleanOptionalAction, default=False,
                   help='train class 0 only; output class 1 as the image-level background complement')
    p.add_argument('--black_threshold', type=int, default=3,
                   help='0-255 intensity threshold for non-black foreground region')
    p.add_argument('--grad_checkpoint', action=argparse.BooleanOptionalAction,
                   default=True, help='activation checkpointing in ViT blocks')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)

    # Optimisation
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=8, help='images per GPU')
    p.add_argument('--accum', type=int, default=1, help='gradient accumulation steps')
    p.add_argument('--lr_backbone', type=float, default=1e-5)
    p.add_argument('--lr_head', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--layer_decay', type=float, default=0.75)
    p.add_argument('--warmup_epochs', type=float, default=0.5)
    p.add_argument('--min_lr_ratio', type=float, default=0.01)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--reg_weight', type=float, default=1.0, help='GIoU loss weight')
    p.add_argument('--train_minutes', type=float, default=0,
                   help='training time budget in minutes; 0 = complete all epochs')

    # Eval / IO
    p.add_argument('--val_batch', type=int, default=8)
    p.add_argument('--val_every', type=int, default=1, help='0 disables mid-training eval')
    p.add_argument('--val_max_images', type=int, default=0, help='0 = full val set')
    p.add_argument('--final_val_max_images', type=int, default=0)
    p.add_argument('--rank_zero_timeout', type=float, default=86400,
                   help='seconds allowed for validation gathering and checkpoint IO (default: 24 hours)')
    p.add_argument('--conf_thres', type=float, default=0.25)
    p.add_argument('--selection_metric', choices=['mAP50_95', 'lesion_AP50_95'],
                   default='lesion_AP50_95', help='validation metric used to select best.pt')
    p.add_argument('--iou_thres', type=float, default=0.7)
    p.add_argument('--save_period', type=int, default=0, help='epochs between snapshots')
    p.add_argument('--save_min_score', type=float, default=-1,
                   help='save best.pt only above this full-validation selection score')
    p.add_argument('--resume', type=str, default='', help='checkpoint to resume from')
    p.add_argument('--init_weights', type=str, default='',
                   help='load detector model weights only; start a new optimizer and LR schedule')

    # Misc
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    run_started = time.monotonic()
    args = parse_args()
    if args.selection_metric == 'lesion_AP50_95' and not args.lesion_only:
        raise ValueError('--selection_metric lesion_AP50_95 requires --lesion_only')
    if args.resume and args.init_weights:
        raise ValueError('--resume and --init_weights are mutually exclusive')
    if not 0 <= args.black_threshold <= 255:
        raise ValueError('--black_threshold must be in [0, 255]')
    if not math.isfinite(args.rank_zero_timeout) or args.rank_zero_timeout <= 0:
        raise ValueError('--rank_zero_timeout must be a finite positive number')
    if not math.isfinite(args.train_minutes) or args.train_minutes < 0:
        raise ValueError('--train_minutes must be finite and non-negative')
    rank, world_size, device, is_distributed = setup_distributed(args.device)
    # All ranks create this group in the same order, before rank-only work.
    rank_zero_group = create_rank_zero_group(args.rank_zero_timeout) if is_distributed else None
    log = rank == 0

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_set, val_set, class_names = build_datasets(
        args.data, imgsz=args.imgsz, lesion_only=args.lesion_only,
        black_threshold=args.black_threshold)
    if log:
        print(f'train images {len(train_set)}, val images {len(val_set)}, '
              f'classes {class_names}', flush=True)

    train_sampler = DistributedSampler(train_set, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=args.workers, collate_fn=collate_fn,
        drop_last=True, pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.val_batch, shuffle=False,
        sampler=ValidationSampler(val_set, rank, world_size) if is_distributed else None,
        num_workers=args.workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=args.workers > 0,
    )

    encoder_checkpoint = None if args.init_weights or args.resume else args.ckpt
    if log:
        print(f'building ViT-G detector; model classes {1 if args.lesion_only else len(class_names)}, '
              f'lesion_only={args.lesion_only}', flush=True)
    base_model = VJepaDetector(
        num_classes=1 if args.lesion_only else len(class_names), resolution=args.imgsz,
        checkpoint=encoder_checkpoint, fpn_strides=tuple(args.fpn_strides),
        fpn_dim=args.fpn_dim, head_convs=args.head_convs,
        drop_path=args.drop_path, grad_checkpoint=args.grad_checkpoint,
        out_layers=args.out_layers,
    ).to(device)

    optimizer = build_optimizer(base_model, args, len(base_model.backbone.blocks))
    criterion = FCOSLoss(fpn_strides=tuple(args.fpn_strides),
                         reg_weight=args.reg_weight).to(device)

    start_epoch = 0
    if args.init_weights:
        ckpt = torch.load(args.init_weights, map_location='cpu', weights_only=False)
        load_detector_weights(base_model, ckpt['model'], lesion_only=args.lesion_only)
        del ckpt
        if log:
            print(f'initialized model weights from {args.init_weights}; '
                  'optimizer and LR schedule start fresh', flush=True)
    elif args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if bool(ckpt['args'].get('lesion_only', False)) != args.lesion_only:
            raise ValueError('resume requires the same --lesion_only mode; use --init_weights for a two-class to lesion-only conversion')
        if ((None if ckpt['args'].get('out_layers') is None
             else tuple(ckpt['args']['out_layers']))
                != (None if args.out_layers is None else tuple(args.out_layers))):
            raise ValueError('resume checkpoint and --out_layers must use the same FPN mode; use --init_weights to change modes')
        base_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        del ckpt
        if log:
            print(f'resumed from {args.resume} at epoch {start_epoch}', flush=True)

    model = base_model
    if is_distributed:
        # No find_unused_parameters: the anchors/heads are always connected to
        # the graph (FCOSLoss keeps zero-valued terms on empty images), and the
        # anchor buffers are constants, so they need no broadcast either.
        model = DDP(base_model, device_ids=[device.index],
                    broadcast_buffers=False, find_unused_parameters=False)

    # The final partial accumulation window still performs an optimizer step,
    # so this must be ceil rather than floor.
    steps_per_epoch = max(math.ceil(len(train_loader) / max(args.accum, 1)), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    if log:
        n_train = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f'trainable params {n_train / 1e6:.1f}M | '
              f'{steps_per_epoch} optimizer steps/epoch, '
              f'{total_steps} total, {warmup_steps} warmup', flush=True)

    run_dir = os.path.join(args.output_dir, args.exp_name)
    if log:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    metrics_path = os.path.join(run_dir, 'metrics.jsonl')

    best_map = -1.0
    training_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        started = time.time()
        remaining_seconds = (args.train_minutes * 60 - training_seconds
                             if args.train_minutes else None)
        if remaining_seconds is not None and remaining_seconds <= 15:
            break
        train_metrics, epoch_seconds, timed_out, batches = train_one_epoch(
            model, base_model, train_loader, optimizer, criterion, device,
            epoch, args, steps_per_epoch, total_steps, warmup_steps,
            is_distributed, log, remaining_seconds)
        training_seconds += epoch_seconds

        record = {'epoch': epoch, 'train': train_metrics,
                  'minutes': (time.time() - started) / 60,
                  'train_batches': batches, 'time_budget_reached': timed_out}
        if log:
            print(f'epoch {epoch} done in {record["minutes"]:.1f} min | '
                  f'train loss {train_metrics["loss"]:.4f}', flush=True)

        if args.val_every and (epoch + 1) % args.val_every == 0:
            record['val_partial'], val_predictions = evaluate(
                base_model, val_loader, device, len(class_names), args,
                max_images=args.val_max_images, return_predictions=True,
                group=rank_zero_group)
            if log:
                m = record['val_partial']
                print(f'  val mAP50 {m["mAP50"]:.4f} mAP50-95 {m["mAP50_95"]:.4f} '
                      f'P {m["precision"]:.4f} R {m["recall"]:.4f} F1 {m["f1"]:.4f}',
                      flush=True)

        # Global metrics are broadcast, so every rank makes the same decision
        # and joins the optional full-set export evaluation.
        score = record.get('val_partial', {}).get(args.selection_metric)
        if log and args.lesion_only and 'val_partial' in record:
            m = record['val_partial']
            print(f'  lesion AP50 {m["lesion_AP50"]:.4f} '
                  f'AP50-95 {m["lesion_AP50_95"]:.4f}', flush=True)
        is_best = score is not None and score > best_map
        if is_best:
            best_map = score
            export_metrics = record['val_partial']
            if args.val_max_images:
                export_metrics, val_predictions = evaluate(
                    base_model, val_loader, device, len(class_names), args,
                    return_predictions=True, group=rank_zero_group)

        if log:
            with open(metrics_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

            is_last = epoch == args.epochs - 1
            save_snapshot = args.save_period and (is_last or (epoch + 1) % args.save_period == 0)
            save_best = is_best and score > args.save_min_score
            if save_snapshot or save_best:
                ckpt = {'model': base_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'args': vars(args),
                        'class_names': class_names,
                        'model_num_classes': base_model.head.num_classes}
            if save_snapshot:
                torch.save(ckpt, os.path.join(run_dir, f'last_epoch{epoch}.pt'))

            if save_best:
                torch.save(ckpt, os.path.join(run_dir, 'best.pt'))
                save_predictions(os.path.join(run_dir, 'best_prediction.csv'),
                                 val_predictions, class_names)
                with open(os.path.join(run_dir, 'best_prediction.metrics.json'), 'w') as f:
                    json.dump(dict(checkpoint=os.path.abspath(os.path.join(run_dir, 'best.pt')),
                                   checkpoint_epoch=epoch, class_names=class_names,
                                   data=args.data, imgsz=args.imgsz,
                                   out_layers=args.out_layers,
                                   lesion_only=args.lesion_only,
                                   model_num_classes=base_model.head.num_classes,
                                   black_threshold=args.black_threshold,
                                   conf_thres=args.conf_thres, iou_thres=args.iou_thres,
                                   selection_metric=args.selection_metric,
                                   selection_score=score,
                                   selection_mAP50_95=record['val_partial']['mAP50_95'],
                                   **export_metrics), f, indent=2)
                print(f'  new best {args.selection_metric} {best_map:.4f}', flush=True)
            elif is_best:
                print(f'  run best {args.selection_metric} {best_map:.6f}; '
                      f'checkpoint threshold {args.save_min_score:.6f} not met', flush=True)

        wait_for_rank_zero(rank_zero_group)
        if timed_out:
            break

    if args.final_val_max_images != args.val_max_images:
        final = evaluate(base_model, val_loader, device, len(class_names), args,
                         max_images=args.final_val_max_images, group=rank_zero_group)
    if log and args.final_val_max_images != args.val_max_images:
        print(f'final val mAP50 {final["mAP50"]:.4f} '
              f'mAP50-95 {final["mAP50_95"]:.4f} '
              f'P {final["precision"]:.4f} R {final["recall"]:.4f} '
              f'F1 {final["f1"]:.4f}', flush=True)
        with open(metrics_path, 'a') as f:
            f.write(json.dumps({'final_val': final}) + '\n')

    wait_for_rank_zero(rank_zero_group)
    peak_vram = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**20,
                             device=device)
    if is_distributed:
        dist.all_reduce(peak_vram, op=dist.ReduceOp.MAX)
    if log:
        print('---', flush=True)
        print(f'lesion_AP50_95: {best_map:.9f}', flush=True)
        print(f'training_seconds: {training_seconds:.1f}', flush=True)
        print(f'total_seconds: {time.monotonic() - run_started:.1f}', flush=True)
        print(f'peak_vram_mb: {peak_vram.item():.1f}', flush=True)
    if is_distributed:
        dist.destroy_process_group(rank_zero_group)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
