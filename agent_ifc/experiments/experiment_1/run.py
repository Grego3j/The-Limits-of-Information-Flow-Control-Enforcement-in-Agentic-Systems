"""Experiment 1: 4–20 sensitive tools, original 1,000 20-tool DAGs, five agents."""
import sys as _style_sys
from pathlib import Path as _StylePath
_style_sys.path.insert(0, str(_StylePath(__file__).resolve().parents[3]))
from agent_ifc.experiments.plot_style import topology_color, topology_marker, EXCLUDED_TOPOLOGIES, simplify_figure

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT.parent))
from agent_ifc.reproduction.support import build_topologies
from agent_ifc.reproduction.support import evaluate,realize,reference_labels
from agent_ifc.experiments.experiment_0.run import atomic_json
BUDGETS=list(range(4,21))


def job(pair):
    did=pair['id'];task=pair['task_dag'];tops=build_topologies(5)
    marker=OUT/'checkpoints'/f'dag_{did:04d}.json'
    from agent_ifc.experiments.experiment_2a.run import samples
    path=OUT/'execution_samples'/f'dag_{did:02d}.npz'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        orders,deliveries,info=samples(task,did)
        perm=__import__('random').Random(20260927+did).sample(range(1,21),20)
        np.savez_compressed(path,task_orders=orders,delivery_orders=deliveries,source_permutation=perm)
    sample_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint=hashlib.sha256(json.dumps([task,tops,BUDGETS,sample_hash,'python-regenerated-v1'],sort_keys=True).encode()).hexdigest()
    if marker.exists():
        record=json.loads(marker.read_text());assert record['fingerprint']==fingerprint
        return record
    with np.load(path) as z:
        orders=z['task_orders'];deliveries=z['delivery_orders'];perm=z['source_permutation'].tolist()
    assert orders.shape==(10000,20) and sorted(perm)==list(range(1,21))
    owners={n['id']:n['agent'] for n in task['nodes']}
    ideal=dict(id='ideal',name='Ideal',nodes=[dict(id=f'a{i}') for i in range(1,6)],directed=False,
               edges=sorted({tuple(sorted((owners[u],owners[v]))) for u,v in task['edges'] if owners[u]!=owners[v]}))
    tops=[ideal]+tops
    bits=np.zeros(21,dtype=np.uint32)
    for j,budget in enumerate(BUDGETS):
        for node in perm[:budget]:bits[node]|=np.uint32(1<<j)
    labels=[evaluate(task,orders,deliveries,t,bits) for t in tops]
    sequence=np.bitwise_or.accumulate(bits[orders],axis=1)
    single=np.zeros_like(sequence);np.put_along_axis(single,orders-1,sequence,axis=1)
    labels.append(single)
    assert np.array_equal(labels[0],labels[1])
    for ls in labels:
        assert np.all((ls&labels[0])==labels[0]) and np.all((ls&single)==ls)
        assert np.all((ls&bits[1:])==bits[1:])
    counts=np.array([[((ls>>j)&1).sum(axis=1) for ls in labels] for j in range(17)],dtype=np.uint8).transpose(0,2,1)
    assert np.all(counts[1:].astype(int)>=counts[:-1]) and np.all(counts[-1]==20)
    # Historical native replay comparison is unavailable after archive deletion.
    # Deleted historical samples are unavailable; current replay is checked below.
    scalar_checks=0
    for sample,budget in [(0,4),(5000,12),(9999,20)]:
        dd={t:deliveries[sample,t][deliveries[sample,t]>0].tolist() for t in range(1,21)}
        for j,top in enumerate(tops):
            ref=reference_labels(realize(task,top,orders[sample].tolist(),dd),top,set(perm[:budget]),'execution')
            assert ((labels[j][sample]>>(budget-4))&1).tolist()==[ref[t] for t in range(1,21)]
            scalar_checks+=1
    names=[t['name'] for t in tops]+['Single-agent baseline'];keys=[t['id'] for t in tops]+['single_agent']
    summaries=[]
    for b,budget in enumerate(BUDGETS):
        for j,(key,name) in enumerate(zip(keys,names)):
            values=counts[b,:,j];excess=values.astype(int)-counts[b,:,0]
            summaries.append(dict(budget=budget,topology=key,name=name,propagation=float(values.mean()/20),
                                 excessive_propagation=float(excess.mean()/20),
                                 sensitive_count_histogram=np.bincount(values,minlength=21).tolist(),
                                 excess_count_histogram=np.bincount(excess,minlength=21).tolist()))
    dest=OUT/'samples'/f'dag_{did:04d}.npz';temp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(temp,counts=counts,source_permutation=perm);temp.replace(dest)
    record=dict(dag_id=did,task=task,source_permutation=perm,fingerprint=fingerprint,
                execution_archive=str(path.relative_to(ROOT)),execution_archive_sha256=sample_hash,
                executions=10000,scalar_checks=scalar_checks,original_counts_match=None, execution_samples_regenerated=True,summary=summaries)
    atomic_json(marker,record);return record


def render(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,13))
    keys=list(dict.fromkeys(r['topology'] for r in data['combined'] if r['topology']!='single_agent' and r['topology'] not in EXCLUDED_TOPOLOGIES))
    for j,key in enumerate(keys):
        rows=[r for r in data['combined'] if r['topology']==key]
        for ax,metric in zip(axes,['propagation','excessive_propagation']):
            ax.plot([r['budget'] for r in rows],[r[metric]*100 for r in rows],label=rows[0]['name'].replace('Connected Erdos-Renyi','ER'),
                    color=topology_color(key),marker=topology_marker(key),markersize=7,linestyle='-')
    for ax,metric,title,ylabel in zip(axes,['propagation','excessive_propagation'],
        ['Label Propagation with Increasing Sensitive Tool Count','Excessive Label Propagation with Increasing Sensitive Tool Count'],
        ['Mean label propagation (%)','Mean excessive label propagation (%)']):
        ax.set_title(title.replace(' with Increasing', '\nwith Increasing'),fontsize=23,pad=14);ax.set_xlabel('Number of sensitive tool calls',fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22);ax.set_xticks(BUDGETS);ax.set_xlim(4,20);ax.tick_params(labelsize=20);ax.grid(alpha=.2)
        values=[r[metric]*100 for r in data['combined'] if r['topology'] in keys]
        low,high=min(values),max(values);pad=max(.2,(high-low)*.08);ax.set_ylim(low-pad,high+pad)
    simplify_figure(fig);fig.tight_layout(rect=[0,.17,1,1],h_pad=1.1,pad=.6)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',ncol=3,fontsize=20,
               bbox_to_anchor=(.015,.14,.97,0),mode='expand',borderaxespad=0,
               handlelength=1.5,markerscale=1.5,labelspacing=.35,frameon=False)
    for ext in ['png','svg','pdf']:simplify_figure(fig);fig.savefig(OUT/f'propagation_by_sensitive_tool_count.{ext}',dpi=300,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    for folder in ['checkpoints','samples']:(OUT/folder).mkdir(exist_ok=True)
    input_path=ROOT/'experiments/datasets/dags1000/dags.json'
    pairs=json.loads(input_path.read_text())['dags'];assert len(pairs)==1000
    records=[];started=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(job,pair) for pair in pairs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%25==0:print(f'{len(records)}/1000 DAGs; {time.time()-started:.0f}s',flush=True)
    records.sort(key=lambda r:r['dag_id']);combined=[]
    for i,first in enumerate(records[0]['summary']):
        row={k:first[k] for k in ['budget','topology','name']}
        for metric in ['propagation','excessive_propagation']:
            values=[r['summary'][i][metric] for r in records]
            row[metric]=float(np.mean(values));row[metric+'_dag_sd']=float(np.std(values,ddof=1))
        combined.append(row)
    result=dict(experiment='Experiment 1',dag_count=1000,tools_per_dag=20,agents=5,budgets=BUDGETS,
                executions_per_dag=10000,executions_per_configuration=10000000,routing_policy='root_only_dispatch_v2',
                source_seed='20260927 + DAG ID; reuse saved source permutations',
                methodology='Original graph structures and ownership retained; 10,000 joint task/delivery samples per DAG regenerated with seed 20261002 + DAG ID. Prefixes of the saved uniform permutation select exactly b sensitive tools at budget b; fixed across executions/topologies. Sources activate on execution. Equal DAG weight. Excessive propagation subtracts matched ideal. Nine physical topologies plus ideal and single-agent baselines.',
                input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),topologies=build_topologies(5),combined=combined,per_dag=records)
    atomic_json(OUT/'results.json',result);render(result)
    print('Experiment 1 complete.',flush=True)


if __name__=='__main__':main()
