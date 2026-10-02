"""Plot completed experiment AP/F1 deltas against the fixed reference."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DET = Path(__file__).resolve().parents[1]
OUTPUT = DET / 'output'
REFERENCE = OUTPUT / 'lesion_resolution_640_vs_768_20260928T083805_1134/768/best_prediction.metrics.json'
EARLY_TRAJECTORY = OUTPUT / 'autoresearch_progress_20261001T122853.json'


def main():
    reference = json.loads(REFERENCE.read_text())
    previous = json.loads(EARLY_TRAJECTORY.read_text())
    rows = [{'label': 'Reference', 'ap': reference['lesion_AP50_95'],
             'f1': reference['f1'], 'kind': 'reference'}]
    seen = set()
    for item in previous['rows']:
        if item['label'].startswith('H'):
            rows.append(dict(label=item['label'].split(':')[0], ap=item['ap'],
                             f1=item['f1'], kind='run'))
            seen.add(item['run'])
    for path in OUTPUT.glob('autoresearch_*.result.json'):
        match = re.search(r'autoresearch_(?:sep30|oct1)_h(\d+)_', path.stem)
        if not match:
            continue
        result = json.loads(path.read_text())
        if result.get('status') == 'crash' or result['experiment'] in seen:
            continue
        rows.append(dict(label=f"H{int(match[1])}", ap=result['score'],
                         f1=result['best_metrics']['f1'], kind='run',
                         experiment=result['experiment']))
        seen.add(result['experiment'])
    rows[1:] = sorted(rows[1:], key=lambda row: int(re.search(r'\d+', row['label'])[0]))
    prior_fpn = next(item for item in previous['rows'] if item['label'].startswith('Prior FPN'))
    rows.append(dict(label='Prior FPN', ap=prior_fpn['ap'], f1=prior_fpn['f1'], kind='fpn'))
    for row in rows:
        row['ap_delta_pp'] = (row['ap'] - reference['lesion_AP50_95']) * 100
        row['f1_delta_pp'] = (row['f1'] - reference['f1']) * 100
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    labels = [row['label'] for row in rows[1:]]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True,
                             gridspec_kw={'wspace': .14})
    for ax, field, title, xlim in [
        (axes[0], 'ap_delta_pp', 'A  Primary metric: lesion AP50:95', (-.95, .12)),
        (axes[1], 'f1_delta_pp', 'B  Secondary metric: overall F1', (-.4, .4)),
    ]:
        ax.axvline(0, color='black', lw=.9)
        for i, row in enumerate(rows[1:]):
            color = '#D55E00' if row['kind'] == 'fpn' else '#0072B2'
            value = row[field]
            ax.hlines(i, 0, value, color=color, lw=2, alpha=.75)
            ax.scatter(value, i, color=color, marker='s' if row['kind'] == 'fpn' else 'o',
                       s=36, zorder=3)
            ax.text(value + (.018 if value >= 0 else -.018), i, f'{value:+.3f}',
                    va='center', ha='left' if value >= 0 else 'right', fontsize=8)
        ax.set_xlim(*xlim)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Difference from reference (percentage points)')
        ax.grid(axis='x', alpha=.18)
        ax.invert_yaxis()
    axes[0].set_yticks(y, labels)
    axes[0].set_ylabel('Run (ordered by experiment number)')
    h7 = next(i for i, row in enumerate(rows[1:]) if row['label'] == 'H7')
    axes[0].annotate('H7 was 0.003 pp below reference',
                     xy=(rows[h7 + 1]['ap_delta_pp'], h7), xytext=(-.67, max(-.6, h7 - 1.2)),
                     arrowprops={'arrowstyle': '->', 'lw': .8}, fontsize=8)
    fig.suptitle('Detector experiment trajectory through completed runs', y=.98, fontsize=13)
    fig.text(.02, .025,
             'All completed H-runs used full 5,000-image validation; one exploratory run per recipe, no uncertainty bars. Prior FPN changed multiple training settings.',
             fontsize=9)
    fig.subplots_adjust(left=.13, right=.98, bottom=.10, top=.92)
    stem = OUTPUT / 'autoresearch_trajectory_latest'
    for ext in ('png', 'svg'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=180)
    stem.with_suffix('.json').write_text(json.dumps(
        dict(reference_ap=reference['lesion_AP50_95'], reference_f1=reference['f1'], rows=rows),
        indent=2) + '\n')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
