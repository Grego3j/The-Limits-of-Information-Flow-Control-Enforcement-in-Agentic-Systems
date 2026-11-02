"""Regenerate the five evaluation figures referenced by the paper's LaTeX."""
import json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agent_ifc.topologies.stats import topology_stats
from agent_ifc.experiments.plot_style import topology_color, topology_marker, agent_marker_area, EXCLUDED_TOPOLOGIES
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'figures'
def read(n): return json.loads((ROOT/'experiments'/f'experiment_{n}'/'results.json').read_text())
def save(fig,stem):
    for ext in ('pdf','png','svg'): fig.savefig(OUT/f'{stem}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
def ranks(x):
    """Average ranks for ties (Spearman convention)."""
    x=np.round(np.asarray(x,dtype=float),12);order=np.argsort(x,kind='stable');out=np.empty(len(x),float)
    start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        out[order[start:end]]=(start+end-1)/2+1
        start=end
    return out


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if np.ptp(x)<1e-12 or np.ptp(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])



def main():
    OUT.mkdir(exist_ok=True)
    from agent_ifc.experiments.experiment_0 import run as zero
    from agent_ifc.experiments.experiment_1 import run as one
    zero.render(read('0'),OUT)
    one.OUT=OUT; one.render(read('1'))
    from agent_ifc.experiments.experiment_3a import plot_comparison_agents_sensitive_10 as comparison
    comparison.main()
    for ext in ('pdf','png','svg'):
        shutil.copy2(comparison.HERE/'plots/comparison_agents_sensitive_10'/f'propagation_by_agents.{ext}',OUT/f'propagation_by_agents.{ext}')
    data=read('1'); stats=[topology_stats(t,5) for t in data['topologies']]
    assert len(stats)==9
    features=[('density','Density'),('average_shortest_path_length','Average shortest-path length'),('diameter','Diameter'),('mean_betweenness_centrality','Average betweenness centrality'),('clustering_coefficient','Clustering coefficient')]
    lookup={(r['budget'],r['topology']):r for r in data['combined']}
    matrix=np.array([[correlation(ranks([t[k] for t in stats]),ranks([100*lookup[b,t['topology']]['propagation'] for t in stats])) for b in range(4,11)] for k,_ in features])
    fig,ax=plt.subplots(figsize=(11,3))
    im=ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    ax.set_yticks(range(5),[label for _,label in features]);ax.set_xticks(range(7),range(4,11));ax.set_xlabel('Number of sensitive tool calls')
    for i in range(5):
        for j in range(7):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.6 else 'black')
    fig.colorbar(im,ax=ax,label='Correlation with Utility Loss')
    save(fig,'correlation_heatmaps_both_sensitive_04_10')
    (OUT/'correlations.json').write_text(json.dumps(dict(features=[k for k,_ in features],budgets=list(range(4,11)),spearman=matrix.tolist()),indent=2)+'\n')
    data=read('2a');fig,ax=plt.subplots(figsize=(10,7));edge_rows=[]
    for key in data['topology_order']:
        if key in {'ideal','single_agent'} or key in EXCLUDED_TOPOLOGIES:continue
        rows=sorted([r for r in data['combined'] if r['budget']==10 and r['topology']==key],key=lambda r:r['agent_count'])
        edges=[topology_stats(next(t for t in data['topologies'][str(r['agent_count'])] if t['id']==key),r['agent_count'])['edges'] for r in rows]
        ax.plot(edges,[100*r['propagation'] for r in rows],color=topology_color(key),label=rows[0]['name'].replace('Connected Erdos-Renyi','ER'))
        ax.scatter(edges,[100*r['propagation'] for r in rows],s=[agent_marker_area(r['agent_count']) for r in rows],marker=topology_marker(key),color=topology_color(key))
        edge_rows.extend(dict(topology=key,agents=r['agent_count'],edges=m,utility_loss=r['propagation']) for r,m in zip(rows,edges))
    ax.set_xscale('log');ax.set_xlabel('Security Enforcement Cost (edges, log scale)');ax.set_ylabel('Utility Loss (%)');ax.grid(alpha=.2)
    for n in (5,10,20):ax.scatter([],[],s=agent_marker_area(n),color='gray',label=f'{n} agents')
    ax.legend(ncol=2);save(fig,'edges_vs_propagation')
    (OUT/'edge_counts.json').write_text(json.dumps(edge_rows,indent=2)+'\n')
if __name__=='__main__': main()
