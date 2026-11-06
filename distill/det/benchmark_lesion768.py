"""Short four-GPU throughput benchmark for the real lesion768 + 9639 KD pipeline.

Launch with torch.distributed.run --nproc_per_node 4. No production checkpoint is
modified. Includes real image loading, mosaic, teacher forward, DDP backward,
AdamW and EMA. Does not estimate validation/checkpoint/startup time.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import yaml
from ultralytics.data.build import build_dataloader
from ultralytics.utils.torch_utils import autocast
from distill.det.com import ComparisonTrainer, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--batches', type=int, nargs='+', default=[64,128,256,384,512])
    p.add_argument('--workers', type=int, nargs='+', default=[2,4,8,12,16])
    args = p.parse_args()
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    output = args.output.resolve()
    config = dict(teacher='vjepa', teacher_ckpt=str(ROOT/'ckpts/vjepa_full/best_vjepa_model9639(paper).pt'),
        baseline_root='/home/lx/baselines', teacher_res=224, teacher_batch=4, hook_layer=6,
        alpha=1., beta=.5, seed=0, nms_backend='torchvision', lesion_only=True)
    os.environ['DET_COM_CONFIG'] = json.dumps(config)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        root = ROOT/'distill/det/output/lesion_only_dataset'
        images = sorted((root/'images/train').glob('*'))
        # Uniformly sample throughout the dataset; do not resolve symlinks, which
        # would redirect label lookup to the original two-class annotations.
        count = min(len(images), max(8192, max(args.batches)*(args.warmup+args.steps+2)))
        paths = [str(images[i*len(images)//count]) for i in range(count)]
        (output/'train.txt').write_text('\n'.join(paths)+'\n')
        val = sorted((root/'images/val').glob('*'))[:4]
        (output/'val.txt').write_text('\n'.join(map(str,val))+'\n')
        (output/'data.yaml').write_text(yaml.safe_dump(dict(path=str(output),train='train.txt',val='val.txt',names={0:'lesion'})))
    # Ranks initially need no process-group communication: directory creation is
    # followed by the trainer's distributed dataset guard after group setup.
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    dist.barrier()
    trainer = ComparisonTrainer(overrides=dict(data=str(output/'data.yaml'),
        model=str(ROOT/'distill/det/output/ckpts/ft-scratch.pt'), device=','.join(map(str,range(world))),
        batch=128, imgsz=768, workers=4, epochs=20, optimizer='AdamW', lr0=.0001,
        warmup_bias_lr=0., seed=0, deterministic=True, amp=True,
        project=str(output), name='setup', exist_ok=True, plots=False, save=False,
        val=False, nbs=128, cos_lr=True, close_mosaic=10, compile=False))
    trainer.device = torch.device('cuda',rank)
    trainer._setup_train()
    trainer._model_train()
    dataset = trainer.train_loader.dataset
    trainer.train_loader.__del__()
    trainer.test_loader.__del__()
    trainer.train_loader = None
    trainer.test_loader = None
    reports = []
    def run(global_batch, workers, phase):
        gc.collect(); torch.cuda.empty_cache()
        free = torch.tensor(torch.cuda.mem_get_info()[0], device=trainer.device, dtype=torch.float64)
        dist.all_reduce(free, op=dist.ReduceOp.MIN)
        # Conservative projection from measured peak; avoid an OOM on the card
        # that has another resident process. All ranks take the same decision.
        previous = [r for r in reports if r.get('status')=='ok' and r['phase']=='batch']
        if previous:
            last = previous[-1]
            estimated = last['peak_allocated_GiB'] * global_batch/last['batch'] * 1.15
            available = torch.tensor(torch.cuda.mem_get_info()[0] + torch.cuda.memory_allocated(),
                                     device=trainer.device, dtype=torch.float64)
            dist.all_reduce(available, op=dist.ReduceOp.MIN)
            total_available = available.item()/2**30
            if estimated > total_available - 5:
                if rank==0: print(f'SKIP batch={global_batch}: conservative memory projection {estimated:.1f} GiB',flush=True)
                return None
        loader = build_dataloader(dataset, batch=global_batch//world, workers=workers, rank=rank, shuffle=True)
        loader.sampler.set_epoch(0)
        iterator = iter(loader)
        trainer.optimizer.zero_grad(set_to_none=True)
        elapsed = []
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        for step in range(args.warmup+args.steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            batch = next(iterator)
            with autocast(trainer.amp):
                batch = trainer.preprocess_batch(batch)
                loss, items = trainer.model(batch)
                loss = loss.sum()*world
            trainer.scaler.scale(loss).backward()
            trainer.optimizer_step()
            torch.cuda.synchronize()
            if step>=args.warmup:
                elapsed.append(time.perf_counter()-start)
        times = torch.tensor([sum(elapsed), torch.cuda.max_memory_allocated()/2**30,
                              torch.cuda.max_memory_reserved()/2**30],device=trainer.device)
        dist.all_reduce(times,op=dist.ReduceOp.MAX)
        row = dict(status='ok',phase=phase,batch=global_batch,per_gpu_batch=global_batch//world,
            workers=workers,total_train_workers=loader.num_workers*world,
            steps=args.steps,warmup=args.warmup,seconds_per_step=times[0].item()/args.steps,
            images_per_second=global_batch*args.steps/times[0].item(),
            peak_allocated_GiB=times[1].item(),peak_reserved_GiB=times[2].item())
        reports.append(row)
        if rank==0:
            print('RESULT '+json.dumps(row),flush=True)
            (output/'results.json').write_text(json.dumps(reports,indent=2))
        del batch, loss, items, iterator
        loader.__del__(); del loader
        dist.barrier()
        return row
    try:
        for batch in args.batches:
            run(batch,8,'batch')
        best = max(reports,key=lambda row:row['images_per_second'])
        for workers in args.workers:
            if workers!=8:
                run(best['batch'],workers,'workers')
        # Repeat both the original settings and winner to expose timing noise.
        best = max(reports,key=lambda row:row['images_per_second'])
        run(128,8,'baseline_repeat')
        run(best['batch'],best['workers'],'winner_repeat')
        if rank==0:
            (output/'environment.json').write_text(json.dumps(dict(world_size=world,imgsz=768,
                teacher=config,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                pipeline='real data+mosaic+teacher+DDP backward+AdamW+EMA; excludes initialization/validation/checkpoint',
                sample_count=len(dataset)),indent=2))
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
