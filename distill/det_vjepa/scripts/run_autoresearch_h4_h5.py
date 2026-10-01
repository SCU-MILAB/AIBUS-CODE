"""Collect the running H4 and then execute the preregistered FPN experiment."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
DET = ROOT / 'distill/det_vjepa'
OUTPUT = DET / 'output'
REFERENCE = OUTPUT / 'lesion_resolution_640_vs_768_20260928T083805_1134/768/best.pt'
REFERENCE_SCORE = 0.8104124888777733
H4 = 'autoresearch_sep30_h4_headonly_cos2'
H5 = 'autoresearch_oct1_h5_fpn_half'


def report(message):
    print(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {message}', flush=True)


def running(pid, exp):
    try:
        # PID reuse after reboot must not make an unrelated process our dependency.
        return exp in Path(f'/proc/{pid}/cmdline').read_text()
    except FileNotFoundError:
        return False


def collect(exp, commit, gate, description):
    log = (OUTPUT / f'{exp}.log').read_text()
    values = dict(re.findall(r'^(lesion_AP50_95|training_seconds|peak_vram_mb):\s+([0-9.]+)', log, re.M))
    row = [commit, '0.000000', '0.0', 'crash', description]
    result = {'experiment': exp, 'commit': commit, 'status': 'crash'}
    if len(values) == 3:
        records = [json.loads(line) for line in (OUTPUT / exp / 'metrics.jsonl').read_text().splitlines()]
        best = max(records, key=lambda record: record['val_partial']['lesion_AP50_95'])
        score = best['val_partial']['lesion_AP50_95']
        seconds = float(values['training_seconds'])
        assert all(record['val_partial']['num_images'] == 5000 for record in records)
        assert records[-1]['time_budget_reached'] and 7180 <= seconds <= 7210
        assert abs(score - float(values['lesion_AP50_95'])) < 1e-8
        keep = score > gate
        assert (OUTPUT / exp / 'best.pt').exists() == keep
        row[1:4] = [f'{score:.6f}', f'{float(values["peak_vram_mb"]) / 1024:.1f}', 'keep' if keep else 'discard']
        result.update(status=row[3], score=score, training_seconds=seconds,
                      best_epoch=best['epoch'], best_metrics=best['val_partial'],
                      last_metrics=records[-1]['val_partial'],
                      train_batches=sum(record['train_batches'] for record in records))
    else:
        result['log_tail'] = log.splitlines()[-40:]
    # A run-specific receipt prevents repeated collectors from duplicating a row.
    receipt = OUTPUT / f'{exp}.result.json'
    if not receipt.exists():
        with (DET / 'results.tsv').open('a') as file:
            file.write('\t'.join(row) + '\n')
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    report(json.dumps({key: result[key] for key in ('experiment', 'status', 'score') if key in result}))
    return result


def ready():
    try:
        free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL, text=True)
        amounts = [int(line) for line in free.splitlines()]
        return len(amounts) == 4 and min(amounts) >= 16000
    except (subprocess.CalledProcessError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h4_pid', type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    while running(args.h4_pid, H4):
        time.sleep(50)
    h4 = collect(H4, '8cf9341', REFERENCE_SCORE, 'shorten frozen-backbone cosine horizon to 2 epochs; 120 min')
    gate = h4.get('score', REFERENCE_SCORE) if h4['status'] == 'keep' else REFERENCE_SCORE
    weights = OUTPUT / H4 / 'best.pt' if h4['status'] == 'keep' else REFERENCE
    while not ready():
        report('waiting for four usable GPUs before H5')
        time.sleep(50)
    data = Path('/home/lx/dataset/6cls_detection/data-2cls-remap.yaml')
    assert data.is_file() and weights.is_file()
    assert not (OUTPUT / H5).exists() and not (OUTPUT / f'{H5}.log').exists()
    assert os.statvfs(OUTPUT).f_bavail * os.statvfs(OUTPUT).f_frsize >= 20 * 1024**3
    commit = subprocess.check_output(['git', 'rev-parse', '--short=7', 'HEAD'], text=True).strip()
    command = ['torchrun', '--master_port=12466', '--nproc_per_node=4', '-m', 'distill.det_vjepa.train',
               '--data', str(data), '--init_weights', str(weights), '--output_dir', str(OUTPUT), '--exp_name', H5,
               '--lesion_only', '--black_threshold', '3', '--selection_metric', 'lesion_AP50_95',
               '--imgsz', '768', '--batch', '4', '--accum', '4', '--val_batch', '4',
               '--epochs', '100', '--cosine_epochs', '2', '--lr_backbone', '2e-6', '--lr_head', '2e-5',
               '--drop_path', '0.1', '--layer_decay', '0.85', '--weight_decay', '0.05',
               '--warmup_epochs', '0.1', '--min_lr_ratio', '0.01', '--clip_grad', '1.0', '--reg_weight', '1.0',
               '--val_every', '1', '--val_max_images', '0', '--final_val_max_images', '0',
               '--conf_thres', '0.25', '--iou_thres', '0.7', '--save_period', '0', '--workers', '4', '--seed', '0',
               '--train_minutes', '120', '--save_min_score', str(gate), '--freeze_backbone', '--fpn_topdown_scale', '0.5']
    (OUTPUT / f'{H5}.launch.json').write_text(json.dumps(dict(commit=commit, command=command), indent=2) + '\n')
    report(f'starting {H5}; initialization={weights}; AP checkpoint gate={gate}')
    with (OUTPUT / f'{H5}.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'))
        (OUTPUT / f'{H5}.pid').write_text(str(process.pid) + '\n')
        process.wait()
    collect(H5, commit, gate, 'halve top-down FPN fusion at 768 with frozen backbone; 120 min')


if __name__ == '__main__':
    main()
