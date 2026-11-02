"""Stable topology colors shared by all experiment figures.

Colors encode topology identity, never ordering or experiment number.
Continuous heatmap colors encode values and retain their existing scales.
"""
TOPOLOGY_COLORS = {
    'mesh': '#1f77b4',
    'star': '#ff7f0e',
    'ring': '#2ca02c',
    'hierarchy': '#d62728',
    'random_connected': '#9467bd',
    'random_connected_p01': '#8c564b',
    'random_connected_p02': '#e377c2',
    'random_connected_p03': '#7f7f7f',
    'random_connected_p04': '#bcbd22',
    'ideal': '#17becf',
    'single_agent': '#222222',
}


def topology_color(topology):
    return TOPOLOGY_COLORS[topology]

EXCLUDED_TOPOLOGIES = {'random_connected_p01', 'random_connected_p02', 'random_connected_p04'}
TOPOLOGY_MARKERS = {
    'ideal': 'p', 'mesh': 'h', 'star': '*', 'ring': 'o', 'hierarchy': 'D',
    'random_connected': 'v', 'random_connected_p01': '^',
    'random_connected_p02': '<', 'random_connected_p03': '^',
    'random_connected_p04': '1', 'single_agent': 's',
}

def topology_marker(topology):
    return TOPOLOGY_MARKERS[topology]


def agent_marker_area(agent_count):
    """Scatter area in points squared; 5/10/20 agents have 5/9/17 pt diameters."""
    return (5.0 + 0.8 * (agent_count - 5)) ** 2


def simplify_figure(fig):
    """Remove figure/panel titles and redundant x axes above matching panels."""
    place_legends_inside(fig)
    from matplotlib.text import Text
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection
    # Apply only to plotted curves and legend samples, leaving axes/grids light.
    curves=[line for ax in fig.axes for line in ax.lines]
    legends=list(fig.legends)
    for ax in fig.axes:
        legends.extend(child for child in ax.get_children()
                       if child.__class__.__name__ == 'Legend')
    for legend in legends:
        curves.extend(legend.findobj(match=Line2D))
    for line in curves:
        line.set_linewidth(max(line.get_linewidth(), 3.0))
        if line.get_marker() not in (None, '', 'None', ' '):
            line.set_markeredgewidth(max(line.get_markeredgewidth(), 1.8))
    for ax in fig.axes:
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_linewidths([max(float(w), 1.8) for w in collection.get_linewidths()])
    for text in fig.findobj(match=Text):
        labels={'mean label propagation (%)':'Utility Loss (%)',
                'mean propagation (%)':'Utility Loss (%)',
                'mean excessive label propagation (%)':'Excess Utility Loss (%)',
                'number of edges (log scale)':'Security Enforcement Cost (log scale)',
                'edges per agent (log scale)':'Security Enforcement Cost per Agent (log scale)'}
        if text.get_text().lower() in labels:
            text.set_text(labels[text.get_text().lower()])
        if text.get_text() == 'Ideal':
            text.set_text(r'$G_{\min}$')
    if fig._suptitle is not None:
        fig._suptitle.set_text('')
    axes=[ax for ax in fig.axes if ax.get_visible() and ax.axison and ax.get_label() != '<colorbar>']
    for ax in axes:
        ax.set_title('')
        a=ax.get_position()
        for lower in axes:
            b=lower.get_position()
            aligned=abs(a.x0-b.x0)<.02 and abs(a.width-b.width)<.02 and b.y1<a.y0
            shared=ax.get_shared_x_axes().joined(ax,lower)
            matching=ax.get_xlabel() and ax.get_xlabel()==lower.get_xlabel() and ax.get_xlim()==lower.get_xlim()
            if aligned and (shared or matching):
                ax.set_xlabel('')
                ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False,labeltop=False)
                ax.spines['bottom'].set_visible(False)
                break


def place_legends_inside(fig):
    """Move external keys into data axes; automatic placement minimizes overlap."""
    from matplotlib.legend import Legend
    data_axes=[ax for ax in fig.axes if ax.axison and ax.get_label() != '<colorbar>'
               and (ax.lines or ax.collections or ax.images)]
    if not data_axes:return
    legends=list(fig.legends)
    for ax in fig.axes:
        legends.extend(c for c in ax.get_children() if isinstance(c,Legend))
    grouped={}
    for legend in legends:
        if getattr(legend,'_inside_standard',False):continue
        ax=legend.axes
        target=ax if ax in data_axes else data_axes[-1]
        # Combine keys for the same plot to prevent separate legends overlapping.
        handles=list(legend.legend_handles)
        labels=[t.get_text() for t in legend.get_texts()]
        if legend.get_title().get_text() in ('Agents','Number of agents'):
            labels=['Agents: '+label for label in labels]
        grouped.setdefault(target,[]).extend(zip(handles,labels))
        legend.remove()
    for ax,entries in grouped.items():
        unique={label:handle for handle,label in entries}
        width=ax.get_position().width*fig.get_figwidth()
        cols=2 if width>=6 and len(unique)>5 else 1
        size=18 if width>=6 else 12
        legend=ax.legend(list(unique.values()),list(unique),loc='best',ncol=cols,
                         fontsize=size,frameon=False,handlelength=1.6,
                         columnspacing=.9,labelspacing=.35,borderaxespad=.6)
        legend._inside_standard=True
    if grouped and fig.get_layout_engine() is None:
        fig.tight_layout(pad=.6,h_pad=.8)
