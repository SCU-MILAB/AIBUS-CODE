"""Continue the detector experiment loop with preregistered single-variable runs."""
import argparse
import datetime
import json
import math
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
H5 = 'autoresearch_oct1_h5_multidepth'
STOP = OUTPUT / 'autoresearch.stop'
BASE_RECIPE = dict(fpn_topdown_scale=1.0, freeze_backbone=True, drop_path=0.1,
                   lr_head=2e-5, reg_weight=1.0, cosine_epochs=2.0, accum=4,
                   out_layers=None, reg_centerness_weighted=False, head_train_scope='all', seed=0, center_radius=1.5)
RATIONALES = {
    'center_radius': 'Restrict training positives to locations nearer the GT center to test high-IoU localization versus coverage; evaluation and decoding are unchanged.',
    'out_layers': 'Route existing intermediate encoder outputs to the neck to test the user-observed FPN improvement under controlled optimization.',
    'fpn_topdown_scale': 'Balance coarse semantic features and fine lateral detail to improve high-IoU localization.',
    'drop_path': 'Test whether stochastic depth during frozen-feature adaptation perturbs neck training.',
    'lr_head': 'Control adaptation of the already strong detection neck and head to reduce validation drift.',
    'reg_weight': 'Balance localization against classification and centerness to improve high-IoU boxes.',
    'reg_centerness_weighted': 'Emphasize regression of positive locations with higher target centerness to improve localization of high-confidence boxes.',
    'cosine_epochs': 'Align learning-rate decay with the fixed training-time budget.',
    'freeze_backbone': 'Compare frozen versus trainable encoder weights from the current best checkpoint under the same time budget; freezing after successful encoder adaptation tests whether subsequent updates cause drift.',
    'accum': 'Test optimizer update frequency while retaining the same per-GPU microbatch and time budget.',
    'head_train_scope': 'Restrict neck/head updates to bbox output convolution and level scales; with frozen backbone this preserves raw classification and centerness scores while adapting localization.',
    'seed': 'Change only the random seed for continuation from the current best checkpoint to assess sensitivity to data order and training randomness; this is not an independent replication of the full winning lineage.',
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def commit_files(message, paths):
    assert git('branch', '--show-current') == 'autoresearch/sep30'
    relative = [str(path.relative_to(ROOT)) for path in paths]
    subprocess.run(['git', 'add', '--', *relative], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', message, '--', *relative], cwd=ROOT, check=True)
    return git('rev-parse', '--short=7', 'HEAD')


def record_result(result):
    path = DET / 'autoloop_log.md'
    old = path.read_text() if path.exists() else '# Detector experiment loop\n'
    marker = f'## {result["experiment"]}\n'
    if marker in old:
        return
    details = '\n' + marker + '\n' + f'- Run commit: {result["commit"]}; status: {result["status"]}\n'
    if result['status'] != 'crash':
        m = result['best_metrics']
        details += (f'- Best lesion AP50:95: {result["score"]:.10f}; epoch: {result["best_epoch"]}\n'
                    f'- Overall F1 at best AP: {m["f1"]:.10f}; lesion precision/recall: '
                    f'{m["per_class_precision"][0]:.10f}/{m["per_class_recall"][0]:.10f}\n'
                    f'- Training: {result["training_seconds"]:.1f} seconds, {result["train_batches"]} batches; '
                    'validation: 5,000 images\n')
    else:
        details += '- Training failed or the fixed-budget/full-validation checks failed; see the run log and receipt.\n'
    path.write_text(old + details)
    commit_files(f'research(results): {result["experiment"]}', [path, DET / 'results.tsv'])


def report(message):
    print(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {message}', flush=True)


def running(pid, exp):
    try:
        # PID reuse after reboot must not make an unrelated process our dependency.
        return exp in Path(f'/proc/{pid}/cmdline').read_text()
    except FileNotFoundError:
        return False


def collect(exp, commit, gate, description):
    receipt = OUTPUT / f'{exp}.result.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    log = (OUTPUT / f'{exp}.log').read_text()
    values = dict(re.findall(r'^(lesion_AP50_95|training_seconds|peak_vram_mb):\s+([0-9.]+)', log, re.M))
    row = [commit, '0.000000', '0.0', 'crash', description]
    result = {'experiment': exp, 'commit': commit, 'status': 'crash'}
    if len(values) == 3:
        try:
            records = [json.loads(line) for line in (OUTPUT / exp / 'metrics.jsonl').read_text().splitlines()]
            best = max(records, key=lambda record: record['val_partial']['lesion_AP50_95'])
            score = best['val_partial']['lesion_AP50_95']
            seconds = float(values['training_seconds'])
            assert all(record['val_partial']['num_images'] == 5000 for record in records), 'validation size'
            budget_met = records[-1]['time_budget_reached'] or seconds >= 7185
            assert budget_met and 7180 <= seconds <= 7210, 'training budget'
            assert all(math.isfinite(record['train']['loss']) for record in records), 'nonfinite loss'
            assert math.isfinite(score) and abs(score - float(values['lesion_AP50_95'])) < 1e-8, 'AP summary'
            keep = score > gate
            assert (OUTPUT / exp / 'best.pt').exists() == keep, 'checkpoint gate'
            row[1:4] = [f'{score:.6f}', f'{float(values["peak_vram_mb"]) / 1024:.1f}', 'keep' if keep else 'discard']
            result.update(status=row[3], score=score, training_seconds=seconds,
                          best_epoch=best['epoch'], best_metrics=best['val_partial'],
                          last_metrics=records[-1]['val_partial'],
                          train_batches=sum(record['train_batches'] for record in records))
        except (AssertionError, ValueError, KeyError, OSError) as error:
            row[1:4] = ['0.000000', '0.0', 'crash']
            result['validation_error'] = str(error)
    else:
        result['log_tail'] = log.splitlines()[-40:]
    # A run-specific receipt prevents repeated collectors from duplicating a row.
    if not receipt.exists():
        with (DET / 'results.tsv').open('a') as file:
            file.write('\t'.join(row) + '\n')
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    report(json.dumps({key: result[key] for key in ('experiment', 'status', 'score') if key in result}))
    return result


def ready(minimum_free_mb=16000):
    try:
        free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL, text=True)
        amounts = [int(line) for line in free.splitlines()]
        if len(amounts) != 4 or min(amounts) < minimum_free_mb:
            return False
        subprocess.run(['python', '-c', 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 4'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'))
        return True
    except (subprocess.CalledProcessError, ValueError):
        return False


def command_for(exp, weights, gate, recipe, port=12466):
    command = ['torchrun', f'--master_port={port}', '--nproc_per_node=4', '-m', 'distill.det_vjepa.train',
               '--data', '/home/lx/dataset/6cls_detection/data-2cls-remap.yaml',
               '--init_weights', str(weights), '--output_dir', str(OUTPUT), '--exp_name', exp,
               '--lesion_only', '--black_threshold', '3', '--selection_metric', 'lesion_AP50_95',
               '--imgsz', '768', '--batch', '4', '--val_batch', '4', '--epochs', '100', '--lr_backbone', '2e-6',
               '--layer_decay', '0.85', '--weight_decay', '0.05', '--warmup_epochs', '0.1', '--min_lr_ratio', '0.01',
               '--clip_grad', '1.0', '--val_every', '1', '--val_max_images', '0', '--final_val_max_images', '0',
               '--conf_thres', '0.25', '--iou_thres', '0.7', '--save_period', '0', '--workers', '4',
               '--seed', str(recipe.get('seed', 0)),
               '--train_minutes', '120', '--save_min_score', str(gate), '--export_epoch_predictions']
    for key, value in recipe.items():
        if key == 'seed':
            continue
        elif isinstance(value, bool):
            command.append(f'--{key}' if value else f'--no-{key}')
        elif key == 'out_layers':
            if value is not None:
                command.extend(['--out_layers', *map(str, value)])
        else:
            command.extend([f'--{key}', str(value)])
    return command


def run(exp, weights, gate, recipe, variable, previous):
    receipt = OUTPUT / f'{exp}.result.json'
    if receipt.exists():
        result = json.loads(receipt.read_text())
        record_result(result)
        return result
    protocol = DET / 'experiments' / exp / 'protocol.json'
    log_path = OUTPUT / f'{exp}.log'
    if protocol.exists() and log_path.exists():
        pid_file = OUTPUT / f'{exp}.pid'
        pid = int(pid_file.read_text()) if pid_file.exists() else 0
        while running(pid, exp):
            time.sleep(50)
        commit = git('log', '-1', '--format=%h', '--abbrev=7', '--', str(protocol.relative_to(ROOT)))
        result = collect(exp, commit, gate, f'{variable} {previous} to {recipe[variable]} at 768; 120 min')
        record_result(result)
        return result
    while not ready(16000 if recipe['freeze_backbone'] else 32000):
        if STOP.exists():
            return None
        report(f'waiting for four usable GPUs before {exp}')
        time.sleep(50)
    if STOP.exists():
        return None
    assert Path('/home/lx/dataset/6cls_detection/data-2cls-remap.yaml').is_file() and weights.is_file()
    assert not (OUTPUT / exp).exists() and not log_path.exists()
    while os.statvfs(OUTPUT).f_bavail * os.statvfs(OUTPUT).f_frsize < 20 * 1024**3:
        if STOP.exists():
            return None
        report('waiting for at least 20 GiB of free disk space')
        time.sleep(50)
    command = command_for(exp, weights, gate, recipe)
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(json.dumps(dict(experiment=exp, kind='exploratory', variable=variable,
                                       previous=previous, proposed=recipe[variable],
                                       rationale=RATIONALES[variable], initialization=str(weights),
                                       checkpoint_gate=gate, primary_metric='maximize lesion_AP50_95',
                                       validation_images=5000, training_minutes=120,
                                       recipe=recipe, command=command), indent=2) + '\n')
    commit = commit_files(f'research(protocol): {exp}', [protocol])
    report(f'starting {exp}; {variable}: {previous} -> {recipe[variable]}; checkpoint gate={gate}')
    with log_path.open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'))
        (OUTPUT / f'{exp}.pid').write_text(str(process.pid) + '\n')
        process.wait()
    description = (f'bbox-only near-threshold repeat with seed {recipe[variable]}; 120 min'
                   if variable == 'seed' else
                   f'{variable} {previous} to {recipe[variable]} at 768; 120 min')
    result = collect(exp, commit, gate, description)
    record_result(result)
    return result


def proposals(incumbent):
    # First test the named hypotheses; then refine one coordinate at a time.
    # Revisit the user's routing after a batch-size improvement; unchanged or
    # already tested combinations are skipped by the main loop.
    initial = [('freeze_backbone', True), ('seed', 0), ('seed', 2),
               ('seed', 1), ('center_radius', 1.0), ('center_radius', .75),
               ('center_radius', 1.25), ('accum', 2), ('out_layers', [13, 26, 39]),
               ('head_train_scope', 'box_projection'),
               ('fpn_topdown_scale', .5), ('reg_centerness_weighted', True),
               ('out_layers', [19, 29, 39]), ('fpn_topdown_scale', 0.0),
               ('drop_path', 0.0), ('lr_head', 1e-5), ('reg_weight', 1.5),
               ('fpn_topdown_scale', .75), ('cosine_epochs', 1.0),
               ('freeze_backbone', False), ('reg_weight', 2.0),
               ('lr_head', 5e-6), ('fpn_topdown_scale', .25), ('cosine_epochs', 3.0),
               ('head_train_scope', 'all'),
               ('out_layers', None)]
    yield from initial
    bounds = dict(center_radius=(.5, 2.0), fpn_topdown_scale=(0.0, 1.0), drop_path=(0.0, .2),
                  reg_weight=(.25, 4.0), cosine_epochs=(.2, 12.0))
    widths = dict(center_radius=.25, fpn_topdown_scale=.25, drop_path=.05, reg_weight=.5, cosine_epochs=1.0)
    cycle = 0
    while True:
        for key in ['center_radius', 'lr_head', 'fpn_topdown_scale', 'reg_weight', 'cosine_epochs', 'drop_path']:
            for direction in [-1, 1]:
                if key == 'lr_head':
                    value = min(max(incumbent[key] * 2 ** (direction * 2 ** (-cycle / 2)), 1e-7), 1e-4)
                else:
                    lo, hi = bounds[key]
                    value = min(max(incumbent[key] + direction * widths[key] * 2 ** (-cycle / 2), lo), hi)
                yield key, float(f'{value:.12g}')
        cycle += 1


def value_slug(value):
    """Give repeated proposals of the same recipe field distinct run names."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return '_'.join(map(str, value))
    if value is None:
        return 'none'
    return str(value).replace('-', 'm').replace('.', 'p')


def recover_pending():
    pending = []
    for protocol in sorted((DET / 'experiments').glob('autoresearch_*/protocol.json')):
        saved = json.loads(protocol.read_text())
        exp = saved['experiment']
        receipt = OUTPUT / f'{exp}.result.json'
        if receipt.exists():
            record_result(json.loads(receipt.read_text()))
            continue
        if not (OUTPUT / f'{exp}.log').exists():
            continue
        pid_file = OUTPUT / f'{exp}.pid'
        pid = int(pid_file.read_text()) if pid_file.exists() else 0
        report(f'recovering {exp}, PID {pid}')
        pending.append((protocol, saved, pid))
    # A single collector owns the ledger and git writes for all parallel runs.
    # Collect whichever finishes first instead of waiting behind another run.
    while pending:
        for item in pending.copy():
            protocol, saved, pid = item
            exp = saved['experiment']
            if running(pid, exp):
                continue
            commit = git('log', '-1', '--format=%h', '--abbrev=7', '--', str(protocol.relative_to(ROOT)))
            description = f'{saved["variable"]} {saved["previous"]} to {saved["proposed"]} at 768; 120 min'
            result = collect(exp, commit, saved['checkpoint_gate'], description)
            record_result(result)
            pending.remove(item)
        if pending:
            time.sleep(30)


def known_best():
    gate, weights, recipe = REFERENCE_SCORE, REFERENCE, BASE_RECIPE.copy()
    for receipt in sorted(OUTPUT.glob('autoresearch_*.result.json')):
        result = json.loads(receipt.read_text())
        if result['status'] != 'keep' or result['score'] <= gate:
            continue
        directory = OUTPUT / result['experiment']
        cfg = json.loads((directory / 'config.json').read_text())
        metrics = json.loads((directory / 'best_prediction.metrics.json').read_text())
        assert (directory / 'best.pt').is_file()
        assert metrics['num_images'] == 5000
        assert abs(metrics['lesion_AP50_95'] - result['score']) < 1e-8
        gate, weights = result['score'], directory / 'best.pt'
        recipe = {key: cfg.get(key, default) for key, default in BASE_RECIPE.items()}
    return gate, weights, recipe


def history():
    tested, last_number = set(), 5
    for path in OUTPUT.glob('autoresearch_*_h*'):
        match = re.match(r'autoresearch_[^_]+_h(\d+)_', path.name)
        if match:
            last_number = max(last_number, int(match[1]))
    for protocol in (DET / 'experiments').glob('autoresearch_*/protocol.json'):
        saved = json.loads(protocol.read_text())
        match = re.match(r'autoresearch_[^_]+_h(\d+)_', saved['experiment'])
        if match:
            last_number = max(last_number, int(match[1]))
        if ((OUTPUT / f'{saved["experiment"]}.result.json').exists()
                or (OUTPUT / f'{saved["experiment"]}.log').exists()):
            recipe = dict(BASE_RECIPE, **saved['recipe'])
            tested.add((saved['initialization'], json.dumps(recipe, sort_keys=True)))
    return tested, last_number + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h4_pid', type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    report(f'monitoring existing {H4}, PID {args.h4_pid}; H5 and later single-variable runs will follow')
    while running(args.h4_pid, H4):
        time.sleep(50)
    h4 = collect(H4, '8cf9341', REFERENCE_SCORE, 'shorten frozen-backbone cosine horizon to 2 epochs; 120 min')
    record_result(h4)
    recover_pending()
    gate, weights, incumbent = known_best()
    if not (OUTPUT / f'{H5}.result.json').exists():
        recipe = dict(incumbent, out_layers=[13, 26, 39])
        h5 = run(H5, weights, gate, recipe, 'out_layers', incumbent['out_layers'])
        if h5 is None:
            return
        gate, weights, incumbent = known_best()
    tested, number = history()
    for variable, value in proposals(incumbent):
        if STOP.exists():
            report('manual stop marker found; experiment loop stopped')
            return
        if variable == 'seed':
            # H7 nearly cleared the AP gate. Repeat its bbox-only recipe from
            # the global checkpoint, changing only seed relative to that scope.
            control = dict(incumbent, head_train_scope='box_projection')
            recipe = dict(control, seed=value)
        else:
            control = incumbent
            recipe = dict(incumbent, **{variable: value})
        signature = (str(weights), json.dumps(recipe, sort_keys=True))
        if value == control[variable] or signature in tested:
            continue
        assert sum(recipe[key] != control[key] for key in recipe) == 1
        tested.add(signature)
        exp = f'autoresearch_oct1_h{number}_{variable}_{value_slug(value)}'
        number += 1
        result = run(exp, weights, gate, recipe, variable, control[variable])
        if result is None:
            return
        if result['status'] == 'keep' and result['score'] > gate:
            gate, weights = result['score'], OUTPUT / exp / 'best.pt'
            incumbent.update(recipe)
        report(f'current best lesion AP50:95={gate:.10f}; next experiment will initialize from {weights}')


if __name__ == '__main__':
    main()
