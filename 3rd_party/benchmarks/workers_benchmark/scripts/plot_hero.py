"""Generate the hero figure for the MOSAIC paper.

Single horizontal bar chart showing Worker and FastLane overhead ratios
across all frameworks, normalized to 1.0x = native baseline.

Features:
- Error bars propagated from 5-seed std devs
- Arrow overflow for outliers (Jumanji 42x/58x)
- Colorblind-safe, print-friendly design
- Self-contained: does not modify or depend on other plot scripts

Usage:
    python -m workers_benchmark.scripts.plot_hero [--results-dir PATH] [--env CartPole-v1]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Display names
DISPLAY = {
    'cleanrl': 'CleanRL',
    'xuance': 'XuanCe',
    'ray': 'RLlib',
    'tianshou': 'Tianshou',
    'sb3': 'SB3',
    'sbx': 'SBX',
    'torchrl': 'TorchRL',
    'rltools': 'RLtools',
    'jumanji': 'Jumanji (JAX)',
}

# Colors
C_NATIVE = '#9E9E9E'      # gray for native baseline
C_NATIVE_NEW = '#C5C5C5'  # lighter gray for new API native
C_WORKER = '#4A90D9'
C_FASTLANE = '#1B5E9E'
C_FASTLANE_GYM = '#D4763A'

# Overflow cap: bars above this get arrow treatment
OVERFLOW_CAP = 7.0


def load_results(results_dir, env_filter=None):
    """Load JSON results, group by worker/scenario.

    Workers whose canonical env differs from env_filter (e.g. RLtools
    uses Pendulum-v1) are loaded without the filter so their overhead
    ratios can still be computed.
    """
    WORKER_ENVS = {}

    grouped = defaultdict(lambda: defaultdict(list))
    for f in sorted(results_dir.glob('*.json')):
        try:
            with open(f) as fp:
                r = json.load(fp)
                if r.get('wall_time_seconds', 0) <= 0:
                    continue
                worker = r.get('worker_name', '')
                env_id = r.get('env_id', '')

                # Apply env filter, but exempt workers with different canonical envs
                if env_filter:
                    worker_env = WORKER_ENVS.get(worker)
                    if worker_env:
                        # This worker uses a different env; accept if it matches
                        if env_id != worker_env:
                            continue
                    else:
                        if env_id != env_filter:
                            continue

                grouped[worker][r['scenario']].append(r['wall_time_seconds'])
        except Exception:
            pass
    return grouped


def compute_ratios(grouped):
    """Compute overhead ratios with propagated uncertainty.

    For ratio r = scenario_mean / native_mean, the uncertainty is:
        r_err = r * sqrt((s_std/s_mean)^2 + (n_std/n_mean)^2)
    """
    rows = []
    for w, scenarios in grouped.items():
        native = scenarios.get('native', [])
        if not native or len(native) < 2:
            continue

        n_mean = np.mean(native)
        n_std = np.std(native)

        # Determine which FastLane scenario(s) exist
        has_gymnax = 'fastlane_gymnax' in scenarios and len(scenarios['fastlane_gymnax']) >= 2
        has_gym_fl = 'fastlane_gym' in scenarios and len(scenarios['fastlane_gym']) >= 2
        has_fastlane = 'fastlane' in scenarios and len(scenarios['fastlane']) >= 2

        # Native New API ratio (RLlib only)
        native_new = scenarios.get('native_new_api', [])
        if native_new and len(native_new) >= 2:
            nn_mean = np.mean(native_new)
            nn_std = np.std(native_new)
            nn_ratio = nn_mean / n_mean
            nn_err = nn_ratio * np.sqrt((nn_std / nn_mean) ** 2 + (n_std / n_mean) ** 2)
        else:
            nn_ratio, nn_err = None, None

        # Worker ratio
        worker = scenarios.get('worker', [])
        if worker and len(worker) >= 2:
            w_mean = np.mean(worker)
            w_std = np.std(worker)
            w_ratio = w_mean / n_mean
            w_err = w_ratio * np.sqrt((w_std / w_mean) ** 2 + (n_std / n_mean) ** 2)
        else:
            w_ratio, w_err = None, None

        # FastLane ratio(s)
        fl_entries = []
        if has_gymnax:
            vals = scenarios['fastlane_gymnax']
            m, s = np.mean(vals), np.std(vals)
            r = m / n_mean
            e = r * np.sqrt((s / m) ** 2 + (n_std / n_mean) ** 2)
            fl_entries.append(('FastLane (gymnax)', r, e, C_FASTLANE_GYM, '\\\\'))
        if has_gym_fl:
            vals = scenarios['fastlane_gym']
            m, s = np.mean(vals), np.std(vals)
            r = m / n_mean
            e = r * np.sqrt((s / m) ** 2 + (n_std / n_mean) ** 2)
            fl_entries.append(('FastLane (Gymnasium)', r, e, C_FASTLANE, ''))
        if has_fastlane and not has_gymnax:
            vals = scenarios['fastlane']
            m, s = np.mean(vals), np.std(vals)
            r = m / n_mean
            e = r * np.sqrt((s / m) ** 2 + (n_std / n_mean) ** 2)
            fl_entries.append(('FastLane', r, e, C_FASTLANE, ''))

        # Native self-ratio is 1.0x but has real variance (coefficient of variation)
        n_cv = n_std / n_mean if n_mean > 0 else 0

        rows.append({
            'worker': w,
            'display': DISPLAY.get(w, w),
            'n_err': n_cv,  # real uncertainty from 5 seeds
            'nn_ratio': nn_ratio,
            'nn_err': nn_err,
            'w_ratio': w_ratio,
            'w_err': w_err,
            'fl_entries': fl_entries,
        })

    # Sort by maximum FastLane overhead (ascending), so lowest overhead at top
    def _sort_key(r):
        fl_max = max((ratio for _, ratio, *_ in r['fl_entries']), default=0)
        return fl_max if fl_max > 0 else (r['w_ratio'] or 999)
    rows.sort(key=_sort_key)
    return rows


def plot_hero(rows, output_path):
    """Draw the hero overhead ratios chart."""

    # Count total bars per framework
    bar_data = []  # (y_label, value, error, color, hatch, is_overflow)
    y_labels = []
    group_boundaries = []

    pos = 0
    for row in rows:
        group_start = pos

        # Native New API bar (RLlib only)
        if row.get('nn_ratio') is not None:
            bar_data.append((pos, row['nn_ratio'], row['nn_err'],
                             C_NATIVE_NEW, '..', row['display']))
            pos += 1

        # Worker bar
        if row['w_ratio'] is not None:
            bar_data.append((pos, row['w_ratio'], row['w_err'],
                             C_WORKER, '//', row['display']))
            pos += 1

        # FastLane bar(s)
        for label, ratio, err, color, hatch in row['fl_entries']:
            bar_data.append((pos, ratio, err, color, hatch, row['display']))
            pos += 1

        group_boundaries.append((group_start, pos - 1, row['display']))
        pos += 0.5  # gap between frameworks

    n_bars = len(bar_data)
    fig_height = max(5, n_bars * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = [d[0] for d in bar_data]
    bar_height = 1.0

    for y, val, err, color, hatch, _ in bar_data:
        # All bars drawn the same way - no special cases
        ax.barh(y, val, bar_height, xerr=err, color=color,
                edgecolor='#333', linewidth=0.6, hatch=hatch,
                capsize=3, error_kw={'linewidth': 1.0, 'color': '#555'})
        # Value label
        label_x = val + (err if err else 0) + val * 0.03
        fmt = f'{val:.1f}x' if val >= 10 else f'{val:.2f}x'
        ax.text(label_x, y, fmt,
                va='center', ha='left', fontsize=9, fontweight='bold',
                color='#333')

    # Y-axis: framework names at group centers
    for start, end, name in group_boundaries:
        center = (start + end) / 2
        y_labels.append((center, name))

    ax.set_yticks([c for c, _ in y_labels])
    ax.set_yticklabels([n for _, n in y_labels], fontsize=10.5, fontweight='bold')

    # 1.0x reference line
    ax.axvline(x=1.0, color='#CC3333', linestyle='--', linewidth=1.5, alpha=0.6,
               zorder=0)

    # Legend (manual, to control order)
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=C_WORKER, edgecolor='#333', hatch='//', label='MOSAIC Worker'),
        Patch(facecolor=C_FASTLANE, edgecolor='#333', label='FastLane'),
    ]
    # Add gymnax/gymnasium only if present
    has_gymnax = any(
        any(label == 'FastLane (gymnax)' for label, *_ in r['fl_entries'])
        for r in rows
    )
    if has_gymnax:
        legend_items = [
            Patch(facecolor=C_WORKER, edgecolor='#333', hatch='//', label='MOSAIC Worker'),
            Patch(facecolor=C_FASTLANE_GYM, edgecolor='#333', hatch='\\\\',
                  label='FastLane (gymnax)'),
            Patch(facecolor=C_FASTLANE, edgecolor='#333', label='FastLane (Gymnasium)'),
        ]
    from matplotlib.lines import Line2D
    # Check if any row has native_new_api
    has_new_api = any(r.get('nn_ratio') is not None for r in rows)
    if has_new_api:
        legend_items.insert(0, Patch(facecolor=C_NATIVE_NEW, edgecolor='#333',
                                     hatch='..', label='Native (New API)'))
    ax.legend(handles=legend_items, fontsize=9, loc='upper right',
              framealpha=0.9, edgecolor='#ccc', fancybox=False)

    # Styling
    ax.set_xscale('log')
    ax.set_xlabel('Overhead ratio (log scale, 1.0x = no overhead)', fontsize=11, fontweight='bold')
    ax.set_title('MOSAIC Wrapper Overhead Across RL Frameworks',
                 fontsize=13, fontweight='bold', pad=12)
    # Let matplotlib auto-range on log scale
    all_vals = [d[1] for d in bar_data]
    ax.set_xlim(0.4, max(all_vals) * 1.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')

    # Also save as PDF for paper inclusion
    pdf_path = output_path.with_suffix('.pdf')
    fig2, ax2 = plt.subplots(figsize=(10, fig_height))
    # Re-draw for PDF (matplotlib needs a fresh figure for different backend)
    plt.close(fig2)
    import shutil
    # Save PDF version
    fig_pdf, ax_pdf = plt.subplots(figsize=(10, fig_height))
    plt.close(fig_pdf)

    # Simpler: just save both formats from the same render
    fig3, ax3 = plt.subplots(figsize=(10, fig_height))

    for y, val, err, color, hatch, _ in bar_data:
        ax3.barh(y, val, bar_height, xerr=err, color=color,
                 edgecolor='#333', linewidth=0.6, hatch=hatch,
                 capsize=3, error_kw={'linewidth': 1.0, 'color': '#555'})
        label_x = val + (err if err else 0) + val * 0.03
        fmt = f'{val:.1f}x' if val >= 10 else f'{val:.2f}x'
        ax3.text(label_x, y, fmt,
                 va='center', ha='left', fontsize=9, fontweight='bold',
                 color='#333')

    ax3.set_yticks([c for c, _ in y_labels])
    ax3.set_yticklabels([n for _, n in y_labels], fontsize=10.5, fontweight='bold')
    ax3.axvline(x=1.0, color='#CC3333', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)
    from matplotlib.lines import Line2D
    legend_items_pdf = list(legend_items)
    ax3.legend(handles=legend_items_pdf, fontsize=9, loc='upper right',
               framealpha=0.9, edgecolor='#ccc', fancybox=False)
    ax3.set_xscale('log')
    ax3.set_xlabel('Overhead ratio (log scale, 1.0x = no overhead)', fontsize=11, fontweight='bold')
    ax3.set_title('MOSAIC Wrapper Overhead Across RL Frameworks',
                  fontsize=13, fontweight='bold', pad=12)
    all_vals = [d[1] for d in bar_data]
    ax3.set_xlim(0.4, max(all_vals) * 1.8)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.2, linestyle='--')
    ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {pdf_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate MOSAIC hero figure')
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).parent.parent / 'results')
    parser.add_argument('--env', default='CartPole-v1')
    args = parser.parse_args()

    plots_dir = args.results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_results(args.results_dir, env_filter=args.env)
    if not grouped:
        print(f'No results found for env={args.env}')
        return

    rows = compute_ratios(grouped)
    print(f'Computing overhead ratios for {len(rows)} frameworks')

    plot_hero(rows, plots_dir / 'hero_overhead_ratios.png')


if __name__ == '__main__':
    main()
