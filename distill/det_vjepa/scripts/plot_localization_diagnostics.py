"""Plot descriptive coverage/localization summaries from saved diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', nargs='+', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True,
                        help='one label per diagnostic; use | for a line break')
    parser.add_argument('--output', type=Path, required=True, help='new output stem')
    parser.add_argument('--note', default='Single exploratory runs; no repeated-run uncertainty.')
    args = parser.parse_args()
    assert len(args.labels) == len(args.diagnostics)
    assert args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1])
    assert not any(args.output.with_suffix('.' + ext).exists() for ext in ('png', 'svg', 'json'))
    rows, reference = [], None
    for label, path in zip(args.labels, args.diagnostics):
        data = json.loads(path.read_text())
        assert data['images'] == 5000 and data['lesions'] == 4157
        assert len(data['paired_comparisons']) == 1
        candidate_path, paired = next(iter(data['paired_comparisons'].items()))
        candidate = data['runs'][candidate_path]
        ref = data['runs'][data['paired_reference']]['area_quartiles'][0]['iou_at_least_0.90']
        assert reference is None or reference == ref
        reference = ref
        mutual = paired['mutually_ge_0_50']
        rows.append(dict(label=label.replace('|', '\n'), source=str(path),
                         gain=paired['candidate_only_ge_0_50'], loss=paired['reference_only_ge_0_50'],
                         delta_all=mutual['all']['mean_iou_delta'],
                         delta_small=mutual['area_quartile_1']['mean_iou_delta'],
                         n_all=mutual['all']['n'], n_small=mutual['area_quartile_1']['n'],
                         fraction_small_ge90=candidate['area_quartiles'][0]['iou_at_least_0.90']))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={'wspace': .38})
    x, width = np.arange(len(rows)), .36
    blue, orange = '#0072B2', '#D55E00'
    panels = [([r['gain'] for r in rows], [-r['loss'] for r in rows]),
              ([r['delta_all'] * 1000 for r in rows], [r['delta_small'] * 1000 for r in rows])]
    legends = [('New matches', 'Lost matches'), ('All sizes', 'Smallest quarter')]
    for ax, values, names in zip(axes, panels, legends):
        for offset, value, color, name in zip((-.18, .18), values, (blue, orange), names):
            ax.bar(x + offset, value, width, color=color, label=name)
            for i, v in enumerate(value):
                text = str(int(v)) if ax is axes[0] else f'{v:.2f}'
                padding = .8 if ax is axes[0] else .13
                ax.text(i + offset, v + (padding if v >= 0 else -padding), text,
                        ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
        ax.axhline(0, color='black', lw=.7)
        ax.legend(frameon=False, loc='lower right' if ax is axes[0] else 'lower left', fontsize=9)
    axes[0].set_title('A  Match coverage vs reference', loc='left')
    axes[0].set_ylabel('GT best matches reaching IoU 0.5')
    axes[0].set_ylim(-max(r['loss'] for r in rows) * 1.2 - 1, max(r['gain'] for r in rows) * 1.5 + 1)
    axes[1].set_title('B  Paired localization change', loc='left')
    axes[1].set_ylabel('Mean IoU difference × 1,000\n(mutually reaching IoU 0.5)')
    values = [v for group in panels[1] for v in group]
    axes[1].set_ylim(min(0, min(values)) - .8, max(0, max(values)) + .5)
    fractions = [r['fraction_small_ge90'] * 100 for r in rows]
    axes[2].bar(x, fractions, .6, color=blue)
    axes[2].axhline(reference * 100, color=orange, ls='--', lw=1.3,
                   label=f'Reference {reference * 100:.2f}%')
    axes[2].set_ylim(0, 65)
    axes[2].set_ylabel('Smallest-quarter GT share (%)')
    axes[2].set_title('C  Best-match IoU at least 0.9', loc='left')
    axes[2].legend(frameon=False, loc='upper left', fontsize=9)
    for i, value in enumerate(fractions):
        axes[2].text(i, value + 1, f'{value:.2f}%', ha='center', fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [r['label'] for r in rows], fontsize=9)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Coverage and localization diagnostics: 5,000 images, 4,157 original lesions',
                 y=.99, fontsize=13)
    fig.text(.02, .09, 'Descriptive best matches; no one-to-one assignment. These are not official AP/recall. ' + args.note, fontsize=9)
    fig.text(.02, .04, 'Panel B uses reference-candidate intersections (all/small n): ' +
             '; '.join(f"{r['label'].splitlines()[0]} {r['n_all']}/{r['n_small']}" for r in rows) +
             '. Subsets differ across candidates.', fontsize=9)
    fig.subplots_adjust(bottom=.25, top=.86)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(args.output.with_suffix('.' + ext), dpi=180)
    args.output.with_suffix('.json').write_text(json.dumps(dict(reference_small_fraction_ge90=reference,
                                                               note=args.note, rows=rows), indent=2) + '\n')
    print(args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
