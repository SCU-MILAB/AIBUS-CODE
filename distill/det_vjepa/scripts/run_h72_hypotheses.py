"""Supervise six paired H72 experiments; all start from the same checkpoint."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time

DET = Path(__file__).resolve().parents[1]
ROOT = DET.parents[1]
WEIGHTS = DET / 'output/autoresearch_oct3_h72_reg_weight_0p5/best.pt'
DATA = Path('/home/lx/dataset/6cls_detection/data-2cls-remap.yaml')
GATE = 0.8118909046053886
ROWS = [
    ('control', '0,1', 'all', 'positive', 'none'),
    ('h1_no_ctr', '0,1', 'no_ctr', 'positive', 'none'),
    ('h1_reg_only', '0,1', 'reg_only', 'positive', 'none'),
    ('h3_per_gt', '2,3', 'all', 'gt', 'none'),
    ('h2_token_residual', '2,3', 'all', 'positive', 'token'),
    ('h2_image_residual', '2,3', 'all', 'positive', 'image'),
]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def command(campaign, phase, row, port):
    name, gpu, route, normalization, residual = row
    smoke = phase == 'smoke'
    return [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
            f'--master_port={port}', '-m', 'distill.det_vjepa.train',
            '--data', str(DATA), '--init_weights', str(WEIGHTS),
            '--output_dir', str(campaign / phase), '--exp_name', name,
            '--lesion_only', '--black_threshold', '3', '--selection_metric', 'lesion_AP50_95',
            '--imgsz', '768', '--fpn_strides', '8', '16', '32', '--fpn_dim', '256',
            '--head_convs', '4', '--fpn_topdown_scale', '1', '--head_train_scope', 'box_projection',
            '--no-freeze_backbone', '--drop_path', '0.1', '--grad_checkpoint', '--amp',
            '--batch', '4', '--accum', '8', '--val_batch', '4', '--workers', '2',
            '--epochs', '10', '--cosine_epochs', '10', '--warmup_epochs', '0.1',
            '--train_minutes', '0.5' if smoke else '0',
            '--lr_backbone', '2e-6', '--lr_head', '2e-5', '--layer_decay', '0.85',
            '--weight_decay', '0.05', '--min_lr_ratio', '0.01', '--clip_grad', '1',
            '--center_radius', '1.25', '--reg_weight', '0.5', '--no-reg_centerness_weighted',
            '--reg_l1_weight', '0', '--reg_iou_alpha', '1', '--ctr_target', 'centerness',
            '--backbone_gradient', route, '--reg_normalization', normalization,
            '--box_residual', residual, '--flip_prob', '0.5',
            '--val_every', '0' if smoke else '1', '--val_max_images', '0',
            '--final_val_max_images', '0', '--conf_thres', '0.25', '--iou_thres', '0.7',
            '--save_period', '0', '--save_min_score', str(GATE),
            '--export_epoch_predictions', '--seed', '1', '--sampler_seed', '0',
            '--reset_training_rng', '--log_every', '8' if smoke else '80']


def snapshot(campaign):
    import torch
    source = campaign / 'source'
    source.mkdir()
    hashes = {}
    for relative in ['model.py', 'loss.py', 'train.py', 'infer.py', 'dataset.py',
                     'metric.py', 'evaluation.py', 'distributed.py',
                     'scripts/run_h72_hypotheses.py', 'tests/test_h72_hypotheses.py']:
        path = DET / relative
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative] = digest(path)
    for filename, cmd in [('git_status.txt', ['git', 'status', '--short']),
                          ('git_diff.patch', ['git', 'diff']),
                          ('environment_packages.txt', [sys.executable, '-m', 'pip', 'freeze'])]:
        with (campaign / filename).open('w') as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return dict(source_sha256=hashes, checkpoint=dict(path=str(WEIGHTS), sha256=digest(WEIGHTS)),
                data_yaml=dict(path=str(DATA), sha256=digest(DATA)),
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                platform=platform.platform(),
                gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                             '--format=csv'], text=True))


def progress(run):
    path = Path(run['log'])
    if not path.exists():
        return 'initializing'
    with path.open('rb') as stream:
        stream.seek(max(path.stat().st_size - 24000, 0))
        lines = stream.read().decode(errors='replace').splitlines()
    interesting = [x.strip() for x in lines if 'epoch ' in x or 'lesion AP50' in x
                   or 'Traceback' in x or 'Error' in x or 'initialized model' in x]
    return interesting[-1] if interesting else 'initializing'


def update(campaign, phase, runs):
    public = [{k: v for k, v in r.items() if k != 'process'} for r in runs]
    for row in public:
        row['progress'] = progress(row)
    write_json(campaign / f'{phase}_status.json', dict(updated_at=now(), runs=public))
    lines = ['# H72 hypothesis training', '', f'Updated: {now()}', '',
             f'Phase: {phase}; initialization: {WEIGHTS}', '',
             'All formal runs: 10 epochs, effective batch 64, full 5000-image validation.', '',
             '| Run | GPUs | PID | State | Progress |', '|---|---|---:|---|---|']
    for row in public:
        lines.append(f"| {row['name']} | {row['gpus']} | {row['pid']} | {row['status']} | "
                     f"{row['progress'].replace('|', '/')} |")
    lines += ['', 'Best checkpoints are saved only above H72 AP50–95; every formal epoch exports predictions.',
              'Stop signal: SIGTERM this phase supervisor; it stops only its own active process groups.']
    (campaign / f'{phase}_status.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--phase', choices=['smoke', 'train'], required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        parser.error('tag must be alphanumeric/underscore')
    campaign = DET / 'output' / f'h72_hypotheses_{args.tag}'
    if args.phase == 'smoke':
        campaign.mkdir(exist_ok=False)
        (campaign / 'protocol.md').write_text(
            '# Frozen experiment protocol\n\n'
            'User authorized all hypotheses from H72 best, 10 epochs, concurrent execution.\n'
            'Six rows share seed1/sampler0, 768, effective batch64, original augmentation, '
            'LR2e-6/2e-5, 10-epoch cosine, warmup0.1, fixed full validation.\n'
            'Control is shared by H1/H2/H3; token residual controls extra capacity for image residual.\n'
            'Primary: maximize lesion_AP50_95. Secondary: AP90/AP95, size slices, localization errors.\n'
            'No adaptive incumbent switching. Every row initializes from the same H72.\n'
            'Formal completion requires ten recorded epochs and exit0; smoke timeout is not formal completion.\n'
            'NaN/Inf/OOM/nonzero exit: mark failed, retain artifacts; do not interpret as a negative result.\n'
            'Single paired seed is exploratory; require paired repeats before scientific claims.\n'
            'Shared GPU contention changes wall time; ten full epochs fix sample/update budget.\n'
            'No new augmentation. No evaluation, data split, backbone structure, decoding changes.\n')
        write_json(campaign / 'provenance.json', snapshot(campaign))
    else:
        result = json.loads((campaign / 'smoke_results.json').read_text())
        if not all(r['passed'] for r in result['runs']):
            raise RuntimeError('all six real H72 DDP smoke checks must pass before formal training')
        provenance = json.loads((campaign / 'provenance.json').read_text())
        for path, expected in provenance['source_sha256'].items():
            if digest(DET / path) != expected:
                raise RuntimeError(f'source changed after smoke test: {path}')
    phase_dir = campaign / args.phase
    phase_dir.mkdir(exist_ok=False)
    free = shutil.disk_usage(campaign).free
    if free < 96 * 1024**3:
        raise RuntimeError('need 96 GiB free for six possible best checkpoints plus headroom')
    (campaign / f'{args.phase}_supervisor.pid').write_text(str(os.getpid()) + '\n')
    # Reserve distinct rendezvous ports for all jobs before launching any.
    reservations = []
    for _ in ROWS:
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        reservations.append(sock)
    runs = []
    import signal
    def stop(signum, frame):
        for run in runs:
            if run['process'].poll() is None:
                os.killpg(run['pid'], signal.SIGTERM)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for row, sock in zip(ROWS, reservations):
        name, gpu, route, normalization, residual = row
        port = sock.getsockname()[1]
        argv = command(campaign, args.phase, row, port)
        protocol = dict(name=name, phase=args.phase, frozen_at=now(), initialization=str(WEIGHTS),
                        gpus=gpu, world_size=2, effective_batch=64, epochs=10,
                        intervention=dict(backbone_gradient=route, reg_normalization=normalization,
                                          box_residual=residual), command=argv)
        write_json(phase_dir / f'{name}_protocol.json', protocol)
        sock.close()
        log_path = phase_dir / f'{name}.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
                   PYTHONHASHSEED='1', PYTHONUNBUFFERED='1')
        with log_path.open('x') as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        runs.append(dict(name=name, gpus=gpu, pid=process.pid, started_at=now(), log=str(log_path),
                         status='running', command=argv, process=process))
        print(f'launched {args.phase}/{name}: PID {process.pid}, GPUs {gpu}', flush=True)
        update(campaign, args.phase, runs)
        time.sleep(4)
    while any(r['process'].poll() is None for r in runs):
        for run in runs:
            code = run['process'].poll()
            if code is not None and run['status'] == 'running':
                run.update(exit_code=code, ended_at=now(), status='exited' if code == 0 else 'failed')
        update(campaign, args.phase, runs)
        time.sleep(15)
    results = []
    for run in runs:
        code = run['process'].wait()
        records_path = phase_dir / run['name'] / 'metrics.jsonl'
        records = [json.loads(x) for x in records_path.read_text().splitlines()] if records_path.exists() else []
        epochs = [r for r in records if 'epoch' in r]
        if args.phase == 'smoke':
            import math
            passed = (code == 0 and len(epochs) == 1 and epochs[0]['train_batches'] >= 8
                      and all(math.isfinite(v) for v in epochs[0]['train'].values()))
        else:
            passed = (code == 0 and [r['epoch'] for r in epochs] == list(range(10))
                      and all(r.get('val_partial', {}).get('num_images') == 5000 for r in epochs))
        run.update(exit_code=code, ended_at=now(), status=('smoke_passed' if args.phase == 'smoke'
                                                        else 'completed') if passed else 'failed')
        results.append(dict(name=run['name'], exit_code=code, passed=passed,
                            epoch_count=len(epochs), metrics=str(records_path)))
    update(campaign, args.phase, runs)
    write_json(campaign / f'{args.phase}_results.json', dict(finished_at=now(), runs=results))
    if not all(r['passed'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
