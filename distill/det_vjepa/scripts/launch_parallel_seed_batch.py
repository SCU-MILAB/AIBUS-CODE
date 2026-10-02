"""Launch the preregistered bbox-projection seed repeats alongside H13."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from distill.det_vjepa.scripts import run_autoresearch_h4_h5 as loop


def main():
    os.chdir(loop.ROOT)
    assert loop.git('branch', '--show-current') == 'autoresearch/sep30'
    recipes = [
        ('autoresearch_oct1_h14_seed_1', 1),
        ('autoresearch_oct2_h15_seed_2_parallel', 2),
    ]
    control = dict(loop.BASE_RECIPE, head_train_scope='box_projection')
    gate, weights = loop.REFERENCE_SCORE, loop.REFERENCE
    protocols = []
    commands = []
    for exp, seed in recipes:
        recipe = dict(control, seed=seed)
        protocol = loop.DET / 'experiments' / exp / 'protocol.json'
        log_path = loop.OUTPUT / f'{exp}.log'
        assert not protocol.exists() and not log_path.exists()
        command = loop.command_for(exp, weights, gate, recipe, port=12467 + seed - 1)
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text(json.dumps(dict(
            experiment=exp, kind='exploratory', variable='seed', previous=0,
            proposed=seed, rationale=loop.RATIONALES['seed'],
            initialization=str(weights), checkpoint_gate=gate,
            primary_metric='maximize lesion_AP50_95', validation_images=5000,
            training_minutes=120, recipe=recipe, command=command), indent=2) + '\n')
        protocols.append(protocol)
        commands.append((exp, command))

    commit = loop.commit_files('research(protocol): parallel bbox seed repeats', protocols)
    pids = {}
    for exp, command in commands:
        log_path = loop.OUTPUT / f'{exp}.log'
        with log_path.open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'),
                                       start_new_session=True)
        pids[exp] = process.pid
        (loop.OUTPUT / f'{exp}.pid').write_text(str(process.pid) + '\n')
        loop.report(f'launched {exp}; PID {process.pid}; seed={1 if "h14" in exp else 2}; port={12467 if "h14" in exp else 12468}')

    # The main queue owns H14 collection. Wait for its receipt before this
    # companion writes H15 results, keeping git/result commits serialized.
    exp2 = recipes[1][0]
    pid2 = pids[exp2]
    while loop.running(pid2, exp2):
        time.sleep(50)
    h14_receipt = loop.OUTPUT / f'{recipes[0][0]}.result.json'
    while not h14_receipt.exists():
        time.sleep(10)
    result = loop.collect(exp2, commit, gate, 'bbox-only seed 2 parallel repeat at 768; 120 min')
    loop.record_result(result)
    loop.report(f'parallel seed batch completed: {json.dumps({k: result.get(k) for k in ("experiment", "status", "score")})}')


if __name__ == '__main__':
    main()
