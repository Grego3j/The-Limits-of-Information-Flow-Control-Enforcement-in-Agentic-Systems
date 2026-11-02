"""Overlay Experiments 2a and 3-a at ten sensitive tool calls using saved aggregate results."""
import sys as _style_sys
from pathlib import Path as _StylePath
_style_sys.path.insert(0, str(_StylePath(__file__).resolve().parents[3]))
from agent_ifc.experiments.plot_style import topology_color, topology_marker, EXCLUDED_TOPOLOGIES, simplify_figure

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent


def main():
    data = {e: json.loads((HERE.parent / f'experiment_{e}a' / 'results.json').read_text()) for e in (2, 3)}
    assert data[2]['topologies'] == data[3]['topologies']
    keys = [k for k in data[2]['topology_order'] if k != 'single_agent' and k not in {'random_connected_p01','random_connected_p02','random_connected_p04'}]
    rows = {e: [r for r in data[e]['combined'] if r['budget'] == 10 and r['topology'] in keys] for e in data}
    assert all(len(rs) == len(keys)*16 for rs in rows.values())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    markers = {'ideal': 'p', 'mesh': 'h', 'star': '*', 'ring': 'o',
               'hierarchy': 'D', 'random_connected': 'v',
               'random_connected_p03': '^'}
    handles = []
    for j, key in enumerate(keys):
        color = topology_color(key)
        for e in (2, 3):
            selected = sorted((r for r in rows[e] if r['topology'] == key), key=lambda r:r['agent_count'])
            assert [r['agent_count'] for r in selected] == list(range(5, 21))
            if e == 2:
                handles.append(Line2D([], [], color=color, marker=markers[key], markersize=7, lw=2, label=selected[0]['name'].replace('Connected Erdos-Renyi', 'ER')))
            for ax, metric in zip(axes, ['propagation', 'excessive_propagation']):
                ax.plot([r['agent_count'] for r in selected], [100*r[metric] for r in selected],
                        color=color, marker=markers[key], markersize=7, linewidth=1.7,
                        linestyle='-' if e == 2 else '--', markerfacecolor=color if e == 2 else 'white')
    for ax, metric, title in zip(axes, ['propagation', 'excessive_propagation'], ['Label Propagation', 'Excessive Label Propagation']):
        ax.set_title(f'{title} with 10 sensitive tool calls', fontsize=23, pad=12)
        ax.set_xlabel('Number of agents', fontsize=22)
        ax.set_ylabel('Mean '+title.lower()+' (%)', fontsize=22)
        ax.set_xticks(range(5,21)); ax.set_xlim(5,20); ax.tick_params(labelsize=20); ax.grid(alpha=.2)
        ys = [100*r[metric] for rs in rows.values() for r in rs]
        pad = max(.2, (max(ys)-min(ys))*.08)
        ax.set_ylim(min(ys)-pad, max(ys)+pad)
    handles += [Line2D([], [], color='#444', marker='o', label='Uniform'),
                Line2D([], [], color='#444', marker='o', markerfacecolor='white', linestyle='--', label='Clustered')]
    # Reserve an unobstructed band inside the lower panel for the shared key.
    low, high = axes[1].get_ylim()
    axes[1].set_ylim(low, high + .28 * (high - low))
    legend = axes[1].legend(handles=handles, loc='upper left', ncol=3,
                           fontsize=18, frameon=False,
                           bbox_to_anchor=(.01, .99, .98, 0), mode='expand',
                           borderaxespad=0, handlelength=2.2,
                           markerscale=1.3, labelspacing=.4)
    legend._inside_standard = True
    simplify_figure(fig)
    fig.tight_layout(h_pad=1.1, pad=.6)
    folder = HERE / 'plots' / 'comparison_agents_sensitive_10'
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        simplify_figure(fig);fig.savefig(folder / f'propagation_by_agents.{ext}', dpi=300, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    (folder/'plot_data.json').write_text(json.dumps(rows, indent=2)+'\n')

if __name__ == '__main__':
    main()
