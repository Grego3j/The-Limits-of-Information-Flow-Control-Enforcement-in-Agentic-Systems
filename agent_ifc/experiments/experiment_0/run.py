"""Experiment 0 — five-agent tool-count sweep: one fixed uniformly selected sensitive tool."""
import sys as _style_sys
from pathlib import Path as _StylePath
_style_sys.path.insert(0, str(_StylePath(__file__).resolve().parents[3]))
from agent_ifc.experiments.plot_style import topology_color, topology_marker, EXCLUDED_TOPOLOGIES, simplify_figure

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import random
import sys
import time

import networkx as nx
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT.parent))
from agent_ifc.reproduction.support import OrderSampler
from agent_ifc.reproduction.support import COLORS
from agent_ifc.reproduction.support import build_topologies
from agent_ifc.experiments.experiment_0.engine import evaluate_batch
from agent_ifc.reproduction.support import reference_labels, realize

SEEDS = {'ownership': 20261001, 'sensitive_tool': 20261002, 'orders': 20261003}


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def execution_samples(task, limit, rng):
    sampler = OrderSampler(task)
    joint_count = sampler.total_orders * sampler.total_deliveries
    count = min(joint_count, limit)
    n = len(task['nodes'])
    orders = np.zeros((count, n), dtype=np.int16)
    deliveries = np.full((count, max(sampler.ids)+1, n), -1, dtype=np.int16)
    if joint_count <= limit:
        graph = nx.DiGraph()
        graph.add_nodes_from(sampler.ids)
        graph.add_edges_from(task['edges'])
        choices = [list(permutations(sampler.parent_ids[t])) for t in sampler.ids]
        def enumerate_joint():
            for order in nx.all_topological_sorts(graph):
                for ps in product(*choices):
                    yield order, dict(zip(sampler.ids, ps))
        samples = enumerate_joint()
    else:
        samples = (sampler.sample(rng) for _ in range(limit))
    for i, (order, dd) in enumerate(samples):
        orders[i] = order
        for t, parents in dd.items():
            deliveries[i,t,:len(parents)] = parents
    info = dict(mode='exhaustive' if joint_count <= limit else 'sampled',
                executions=int(count), task_order_count=str(sampler.total_orders),
                delivery_order_count=str(sampler.total_deliveries), joint_order_count=str(joint_count))
    sampler.count.cache_clear()
    return orders, deliveries, info


def job(original, limit, output):
    output = Path(output)
    did = original['id']
    marker = output / 'checkpoints' / f'dag_{did:04d}.json'
    fingerprint = hashlib.sha256(json.dumps([original, limit, SEEDS, build_topologies(5)], sort_keys=True).encode()).hexdigest()
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['fingerprint'] != fingerprint:
            raise ValueError(f'Checkpoint input mismatch for DAG {did}')
        return old
    started = time.time()
    rng = random.Random(SEEDS['ownership'] + did)
    nodes = []
    for node in original['nodes']:
        a = rng.randrange(5)
        nodes.append(dict(node, agent=f'a{a+1}', color=COLORS[a], label=str(node['id'])))
    task = dict(original, nodes=nodes)
    source = random.Random(SEEDS['sensitive_tool'] + did).choice(sorted(n['id'] for n in nodes))
    owners = {n['id']:n['agent'] for n in nodes}
    edges = sorted({tuple(sorted((owners[u], owners[v]))) for u,v in task['edges'] if owners[u] != owners[v]})
    tops = [dict(id='ideal', name='Ideal', directed=False,
                 nodes=[dict(id=f'a{i}') for i in range(1,6)], edges=edges)] + build_topologies(5)
    orders, deliveries, sampling = execution_samples(task, limit, random.Random(SEEDS['orders'] + did))
    labels = [evaluate_batch(task, orders, deliveries, top, {source}, 'execution') for top in tops]
    single = np.zeros((len(orders),len(nodes)), dtype=bool)
    positions = {node['id']:i for i,node in enumerate(nodes)}
    seq = np.maximum.accumulate(orders == source, axis=1)
    for i,node in enumerate(nodes):
        single[:,i] = seq[np.arange(len(orders)), np.argmax(orders == node['id'],axis=1)]
    labels.append(single)
    assert np.array_equal(labels[0],labels[1])
    for ls in labels:
        assert ls[:,positions[source]].all()
        assert np.all(ls | ~labels[0]) and np.all(single | ~ls)
    # Independent scalar route-event cross-check, including every new ER topology.
    for sample in sorted({0,len(orders)//2,len(orders)-1}):
        dd = {node['id']:deliveries[sample,node['id']][deliveries[sample,node['id']]>=0].tolist() for node in nodes}
        for j,top in enumerate(tops):
            ref = reference_labels(realize(task,top,orders[sample].tolist(),dd),top,{source},'execution')
            assert labels[j][sample].tolist() == [ref[node['id']] for node in nodes]
    counts = np.array([ls.sum(axis=1) for ls in labels],dtype=np.uint8).T
    keys = [t['id'] for t in tops] + ['single_agent']
    names = [t['name'] for t in tops] + ['Single-agent baseline']
    n = len(nodes)
    rows = []
    for j,(key,name) in enumerate(zip(keys,names)):
        propagation = counts[:,j].astype(float)/n
        excess = (counts[:,j].astype(int)-counts[:,0])/n
        rows.append(dict(topology=key,name=name,propagation=float(propagation.mean()),
                         excessive_propagation=float(excess.mean()),
                         sensitive_count_histogram=np.bincount(counts[:,j],minlength=n+1).tolist(),
                         excess_count_histogram=np.bincount(counts[:,j].astype(int)-counts[:,0],minlength=n+1).tolist()))
    record = dict(dag_id=did,num_nodes=n,fingerprint=fingerprint,task=task,sensitive_tool=source,
                  sampling=sampling,metrics=rows,seconds=time.time()-started,scalar_checks=len({0,len(orders)//2,len(orders)-1})*len(tops))
    target = output / 'samples' / f'dag_{did:04d}.npz'
    temp = target.with_suffix('.tmp.npz')
    np.savez_compressed(temp,task_orders=orders,delivery_orders=deliveries,sensitive_counts=counts)
    temp.replace(target)
    atomic_json(marker,record)
    return record


def aggregate(records, output, limit):
    combined=[]
    for n in sorted({r['num_nodes'] for r in records}):
        group=[r for r in records if r['num_nodes']==n]
        for j,metric in enumerate(group[0]['metrics']):
            row=dict(num_nodes=n,dag_count=len(group),topology=metric['topology'],name=metric['name'])
            for key in ['propagation','excessive_propagation']:
                values=[r['metrics'][j][key] for r in group]
                row[key]=float(np.mean(values))
                row[key+'_dag_sd']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
            combined.append(row)
    result=dict(experiment='Experiment 0',input_dataset='DAGs/er_only_by_nodes_dags.json',input_sha256=hashlib.sha256((ROOT/'DAGs/er_only_by_nodes_dags.json').read_bytes()).hexdigest(),dag_count=len(records),agent_count=5,maximum_samples=limit,seeds=SEEDS,
                routing_policy='root_only_dispatch_v2',
                methodology='One uniform sensitive tool per DAG, activated on execution; independent uniform ownership over five agents. Both fixed across executions/topologies. Joint task/delivery orders exhaustive if count <= limit, otherwise exact-uniform samples with replacement. Mean within each DAG, then equal-weight mean over DAGs of each size. Excessive propagation = propagation minus matched ideal. All metrics are fractions.',
                exhaustive_dags=sum(r['sampling']['mode']=='exhaustive' for r in records),
                sampled_dags=sum(r['sampling']['mode']=='sampled' for r in records),
                execution_count=sum(r['sampling']['executions'] for r in records),combined=combined,
                per_dag=records,topologies=build_topologies(5))
    atomic_json(output/'results.json',result)
    render(result,output)
    return result


def render(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,13))
    keys=list(dict.fromkeys(r['topology'] for r in data['combined'] if r['topology'] != 'single_agent' and r['topology'] not in EXCLUDED_TOPOLOGIES))
    for j,key in enumerate(keys):
        rows=sorted([r for r in data['combined'] if r['topology']==key],key=lambda r:r['num_nodes'])
        for ax,metric in zip(axes,['propagation','excessive_propagation']):
            ax.plot([r['num_nodes'] for r in rows],[100 * r[metric] for r in rows],label=rows[0]['name'].replace('Connected Erdos-Renyi','ER'),color=topology_color(key),marker=topology_marker(key),markersize=7,linestyle='-')
    for ax,title,ylabel in zip(axes,['Label Propagation with Increasing Task Complexity','Excessive Label Propagation with Increasing Task Complexity'],['Mean label propagation (%)','Mean excessive label propagation (%)']):
        ax.set_title(title.replace(' with Increasing', '\nwith Increasing'),fontsize=23,pad=14);ax.set_xlabel('Number of tool calls',fontsize=22);ax.set_ylabel(ylabel,fontsize=22);ax.set_xticks(range(4,21));ax.set_xlim(4,20);ax.tick_params(axis='both',labelsize=20);ax.grid(alpha=.2)
    for ax,metric in zip(axes,['propagation','excessive_propagation']):
        values=[100 * r[metric] for r in data['combined'] if r['topology'] in keys]
        low,high=min(values),max(values)
        padding=max((high-low)*0.08,0.2)
        ax.set_ylim(low-padding,high+padding)
    handles,labels=axes[0].get_legend_handles_labels()
    simplify_figure(fig);fig.tight_layout(rect=[0,.17,1,1],h_pad=1.1,pad=.6)
    fig.legend(handles,labels,loc='upper left',ncol=3,fontsize=20,
               bbox_to_anchor=(.015,.14,.97,0),mode='expand',borderaxespad=0,
               handlelength=1.5,markerscale=1.5,labelspacing=.35,frameon=False)
    for ext in ['svg','png','pdf']:simplify_figure(fig);fig.savefig(output/f'propagation_by_tool_count.{ext}',dpi=300,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,default=10000)
    parser.add_argument('--first',type=int)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    records=json.loads((ROOT/'DAGs/er_only_by_nodes_dags.json').read_text())['dags']
    if args.first:records=records[:args.first]
    for folder in ['checkpoints','samples']:(args.output/folder).mkdir(parents=True,exist_ok=True)
    started=time.time();completed=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(job,r,args.limit,args.output) for r in records]
        for future in as_completed(futures):
            row=future.result();completed.append(row)
            if len(completed)%25==0:print(f'{len(completed)}/{len(records)} DAGs; {time.time()-started:.0f}s',flush=True)
    result=aggregate(sorted(completed,key=lambda r:(r['num_nodes'],r['dag_id'])),args.output,args.limit)
    print(f"Completed {result['dag_count']} DAGs: {result['exhaustive_dags']} exhaustive, {result['sampled_dags']} sampled.",flush=True)


if __name__=='__main__':main()
