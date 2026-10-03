"""Continuously refill a bounded set of four-GPU research runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from distill.det_vjepa.scripts import run_autoresearch_h4_h5 as core


def discover():
    jobs = []
    for path in sorted((core.DET / 'experiments').glob('autoresearch_*/protocol.json')):
        saved = json.loads(path.read_text())
        name = saved['experiment']
        receipt = core.OUTPUT / f'{name}.result.json'
        if receipt.exists():
            core.record_result(json.loads(receipt.read_text()))
            continue
        if not (core.OUTPUT / f'{name}.log').exists():
            continue
        pid_path = core.OUTPUT / f'{name}.pid'
        pid = int(pid_path.read_text()) if pid_path.exists() else 0
        jobs.append(dict(protocol=path, saved=saved, pid=pid))
    return jobs


def collect_finished(jobs):
    active = []
    for job in jobs:
        saved = job['saved']
        name = saved['experiment']
        if core.running(job['pid'], name):
            active.append(job)
            continue
        commit = core.git('log', '-1', '--format=%h', '--abbrev=7', '--',
                          str(job['protocol'].relative_to(core.ROOT)))
        result = core.collect(name, commit, saved['checkpoint_gate'],
                              f'{saved["variable"]} {saved["previous"]} to {saved["proposed"]} at 768; 120 min')
        core.record_result(result)
    return active


def next_candidate():
    gate, weights, incumbent = core.known_best()
    tested, number = core.history()
    for count, (variable, value) in enumerate(core.proposals(incumbent)):
        if count >= 5000:
            return None
        control = dict(incumbent, head_train_scope='box_projection') if variable == 'seed' else incumbent
        recipe = dict(control, **{variable: value})
        if value == control[variable] or (str(weights), json.dumps(recipe, sort_keys=True)) in tested:
            continue
        assert sum(recipe[key] != control[key] for key in recipe) == 1
        return dict(gate=gate, weights=weights, recipe=recipe, control=control,
                    number=number, variable=variable, value=value)
    return None


def free_port(jobs):
    used = {int(arg.split('=', 1)[1]) for job in jobs
            for arg in job['saved']['command'] if arg.startswith('--master_port=')}
    for port in range(12466, 12496):
        if port in used:
            continue
        try:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
            return port
        except OSError:
            continue
    return None


def launch(candidate, jobs, port):
    recipe, variable = candidate['recipe'], candidate['variable']
    minimum = 16000 if recipe['freeze_backbone'] else 32000
    free_bytes = os.statvfs(core.OUTPUT).f_bavail * os.statvfs(core.OUTPUT).f_frsize
    if free_bytes < (20 + 12 * len(jobs)) * 1024**3 or not core.ready(minimum):
        return None
    weights, gate = candidate['weights'], candidate['gate']
    assert weights.is_file() and Path('/home/lx/dataset/6cls_detection/data-2cls-remap.yaml').is_file()
    name = f'autoresearch_oct3_h{candidate["number"]}_{variable}_{core.value_slug(candidate["value"])}'
    protocol = core.DET / 'experiments' / name / 'protocol.json'
    log_path = core.OUTPUT / f'{name}.log'
    assert not protocol.exists() and not log_path.exists() and not (core.OUTPUT / name).exists()
    command = core.command_for(name, weights, gate, recipe, port=port)
    saved = dict(experiment=name, kind='exploratory', variable=variable,
                 previous=candidate['control'][variable], proposed=candidate['value'],
                 rationale=core.RATIONALES[variable], initialization=str(weights),
                 checkpoint_gate=gate, primary_metric='maximize lesion_AP50_95',
                 validation_images=5000, training_minutes=120, recipe=recipe,
                 control_recipe=candidate['control'], command=command,
                 concurrent_experiments=[job['saved']['experiment'] for job in jobs])
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps(saved, indent=2) + '\n')
    core.commit_files(f'research(protocol): {name}', [protocol])
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=core.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'))
    (core.OUTPUT / f'{name}.pid').write_text(str(process.pid) + '\n')
    core.report(f'launched {name}; PID {process.pid}; gate={gate}; parallel slots={len(jobs)+1}')
    return dict(protocol=protocol, saved=saved, pid=process.pid, process=process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max_parallel', type=int, default=4)
    parser.add_argument('--poll_seconds', type=float, default=30)
    parser.add_argument('--launch_gap_seconds', type=float, default=90)
    args = parser.parse_args()
    assert 1 <= args.max_parallel <= 4 and 0 < args.poll_seconds <= 60
    assert args.launch_gap_seconds >= 60
    os.chdir(core.ROOT)
    assert core.git('branch', '--show-current') == 'autoresearch/sep30'
    with (core.OUTPUT / 'autoresearch_parallel.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid_file = core.OUTPUT / 'autoresearch_h4_h5.queue.pid'
        if pid_file.exists():
            old = int(pid_file.read_text())
            proc = Path(f'/proc/{old}/cmdline')
            if old != os.getpid() and proc.exists():
                command = proc.read_text()
                assert 'run_autoresearch_h4_h5.py' not in command and 'run_parallel_autoresearch' not in command, 'another collector is live'
        pid_file.write_text(str(os.getpid()) + '\n')
        jobs = discover()
        core.report(f'adopted {len(jobs)} unfinished runs; continuous parallel limit={args.max_parallel}')
        next_launch = time.monotonic() + args.launch_gap_seconds
        heartbeat = 0.
        while True:
            for job in jobs:
                if job.get('process') is not None:
                    job['process'].poll()  # reap our completed children
            jobs = collect_finished(jobs)
            if core.STOP.exists() and not jobs:
                core.report('manual stop marker found; all active runs collected')
                return
            if not core.STOP.exists() and len(jobs) < args.max_parallel and time.monotonic() >= next_launch:
                candidate = next_candidate()
                port = free_port(jobs)
                if candidate is not None and port is not None:
                    job = launch(candidate, jobs, port)
                    if job is not None:
                        jobs.append(job)
                        next_launch = time.monotonic() + args.launch_gap_seconds
            if time.monotonic() >= heartbeat:
                core.report('active: ' + ', '.join(f'{job["saved"]["experiment"]}:{job["pid"]}' for job in jobs))
                heartbeat = time.monotonic() + 300
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
