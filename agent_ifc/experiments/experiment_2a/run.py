"""Regenerate paired Experiment 2a / 3-a from immutable Experiment 2 inputs."""
import sys,json,random,math,hashlib,time,argparse
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import deque
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[2]))
from agent_ifc.experiments.experiment_3a import run as core
OLD=OUT.parent/'experiment_2'
OTHER=OUT.parent/'experiment_3a'
REV='paired-agent-first-python-v1'

def samples(task,did):
    ids=sorted(n['id'] for n in task['nodes']);assert ids==list(range(1,21))
    parents={t:sorted(u for u,v in task['edges'] if v==t) for t in ids}
    masks=[sum(1<<(u-1) for u in parents[t]) for t in ids];full=(1<<20)-1
    @lru_cache(None)
    def count(done):
        if done==full:return 1
        return sum(count(done|(1<<i)) for i in range(20) if not done>>i&1 and masks[i]&done==masks[i])
    total=count(0);rng=random.Random(20261002+did)
    orders=np.empty((10000,20),dtype=np.int16);deliveries=np.full((10000,21,20),-1,dtype=np.int16)
    for row in range(10000):
        done=0
        for pos in range(20):
            choices=[i for i in range(20) if not done>>i&1 and masks[i]&done==masks[i]]
            draw=rng.randrange(count(done))
            for i in choices:
                weight=count(done|(1<<i))
                if draw<weight:break
                draw-=weight
            orders[row,pos]=i+1;done|=1<<i
        for t in ids:
            p=parents[t].copy();rng.shuffle(p);deliveries[row,t,:len(p)]=p
    info=dict(seed=20261002+did,task_order_count=str(total),delivery_order_count=str(math.prod(math.factorial(len(p)) for p in parents.values())),
              mode='10000 independent uniform joint samples with replacement',revision=REV)
    return orders,deliveries,info

def reference(task,top,order,delivery,sources):
    owners={n['id']:n['agent'] for n in task['nodes']};adj={n['id']:set() for n in top['nodes']}
    for u,v in top['edges']:adj[u].add(v);adj[v].add(u)
    state=set();outputs=set()
    def route(a,b,payload):
        q=deque([[a]]);seen={a}
        while q:
            path=q.popleft()
            if path[-1]==b:break
            for v in sorted(adj[path[-1]]-seen):seen.add(v);q.append(path+[v])
        for node in path:
            payload=payload or node in state
            if payload:state.add(node)
    for t in order:
        pp=[int(p) for p in delivery[t] if p>0]
        if not pp and 'orchestrator' in adj:route('orchestrator',owners[t],False)
        for p in pp:route(owners[p],owners[t],p in outputs)
        if t in sources:state.add(owners[t])
        if owners[t] in state:outputs.add(int(t))
    return sum(1<<t for t in outputs)

def job(did):
    inputs=core.paired_inputs()['inputs'][str(did)]
    fp=core.digest(json.dumps([inputs,core.TOPS,REV],sort_keys=True).encode())
    markers=[root/'checkpoints'/f'dag_{did:04d}.json' for root in (OUT,OTHER)]
    if all(p.exists() for p in markers):
        rr=[json.loads(p.read_text()) for p in markers]
        if all(r['fingerprint']==fp for r in rr):
            for root,r in zip((OUT,OTHER),rr):assert core.digest((root/'samples'/f'dag_{did:04d}.npz').read_bytes())==r['sample_sha256']
            return rr
        raise ValueError('Existing run fingerprint mismatch')
    archive=OUT/'execution_samples'/f'dag_{did:04d}.npz'
    if archive.exists():
        with np.load(archive) as z:orders=z['task_orders'];deliveries=z['delivery_orders'];info=json.loads(str(z['metadata']))
        assert info['revision']==REV
    else:
        orders,deliveries,info=samples(inputs[0]['task'],did)
        core.save_samples(archive,task_orders=orders,delivery_orders=deliveries,metadata=np.array(json.dumps(info)))
    order_hash=core.digest(orders.tobytes()+deliveries.tobytes())
    # Independently check every saved order and incoming-parent permutation.
    edges=inputs[0]['task']['edges'];positions=np.argsort(orders,axis=1)
    assert all(np.all(positions[:,u-1]<positions[:,v-1]) for u,v in edges)
    for t in range(1,21):
        parents=sorted(u for u,v in edges if v==t)
        assert np.all(np.sort(deliveries[:,t,:len(parents)],axis=1)==np.array(parents,dtype=int))
    payloads=[dict(physical=[],single=[],owners=[],perms=[],propagation=[],excessive_propagation=[],checks=0) for _ in range(2)]
    for k,r in zip(core.AGENTS,inputs):
        task=r['task'];owners=np.zeros(21,dtype=np.uint8)
        for n in task['nodes']:owners[n['id']]=int(n['agent'][1:])
        parents=np.array([0]+[sum(v==t for u,v in task['edges']) for t in range(1,21)],dtype=np.uint8)
        for mode,p in enumerate(payloads):
            perm=r['source_permutation'] if mode==0 else core.source_order(owners,k,did)[0]
            sources=np.array([sum(1<<t for t in perm[:b]) for b in core.BUDGETS],dtype=np.uint32)
            allocations=np.repeat(owners[None,:],17,axis=0)
            physical,masks=core.kernel(k,orders,deliveries,parents,allocations-1,sources)
            single=np.array([np.logical_or.accumulate(np.isin(orders,perm[:b]),axis=1).sum(axis=1) for b in core.BUDGETS],dtype=np.uint8)
            counts=np.concatenate([physical[:,0:1],physical,single[:,None,:]],axis=1).transpose(0,2,1).copy()
            assert np.all(np.diff(counts.astype(int),axis=0)>=0) and np.all(counts[-1]==20)
            assert np.all(counts>=counts[:,:,0,None]) and np.all(counts<=counts[:,:,-1,None])
            assert np.all((masks & sources[:,None,None])==sources[:,None,None])
            for bi in (0,6,16):
                sample=(did*97+k*43+bi*13)%10000
                for j,top in enumerate(core.TOPS[k]):
                    assert reference(task,top,orders[sample],deliveries[sample],set(perm[:core.BUDGETS[bi]]))==int(masks[bi,j,sample]),(did,k,mode,bi,j)
                    p['checks']+=1
            p['physical'].append(counts[:,:,1:10]);p['single'].append(single);p['owners'].append(owners);p['perms'].append(perm)
            p['propagation'].append(counts.mean(axis=1)/20)
            p['excessive_propagation'].append((counts.astype(int)-counts[:,:,0,None]).mean(axis=1)/20)
    results=[]
    for root,p,marker in zip((OUT,OTHER),payloads,markers):
        dest=root/'samples'/f'dag_{did:04d}.npz';tmp=dest.with_suffix('.tmp.npz')
        core.save_samples(tmp,physical_counts=np.array(p['physical']),single_counts=np.array(p['single']),owners=np.array(p['owners']),source_permutations=np.array(p['perms']))
        tmp.replace(dest)
        result=dict(dag_id=did,fingerprint=fp,execution_archive=str(archive.relative_to(core.ROOT)),execution_orders_sha256=order_hash,
            source_permutations=p['perms'],scalar_checks=p['checks'],sample_sha256=core.digest(dest.read_bytes()),
            propagation=np.array(p['propagation']).tolist(),excessive_propagation=np.array(p['excessive_propagation']).tolist())
        core.atomic_json(marker,result);results.append(result)
    return results

def aggregate(records,root,name):
    rows=[]
    for ki,k in enumerate(core.AGENTS):
        for bi,b in enumerate(core.BUDGETS):
            for j,key in enumerate(core.KEYS):
                row=dict(agent_count=k,budget=b,topology=key,name=next(r['name'] for r in core.BASE['combined'] if r['topology']==key))
                for metric in ('propagation','excessive_propagation'):
                    values=[r[metric][ki][bi][j] for r in records];row[metric]=float(np.mean(values));row[metric+'_dag_sd']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
                rows.append(row)
    core.atomic_json(root/'results.json',dict(experiment=name,dag_count=len(records),tools_per_dag=20,agent_counts=core.AGENTS,budgets=core.BUDGETS,
        topology_order=core.KEYS,topologies=core.BASE['topologies'],combined=rows,executions_per_dag=10000,revision=REV,
        scalar_checks=sum(r['scalar_checks'] for r in records),runs=[dict(dag_id=r['dag_id']) for r in records]))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--limit',type=int,default=1000);args=parser.parse_args()
    for root in (OUT,OTHER):
        for name in ('samples','checkpoints','plots'):(root/name).mkdir(exist_ok=True)
    (OUT/'execution_samples').mkdir(exist_ok=True)
    # Pure Python replay; no compiler or native library is required.
    ids=sorted(set(r['dag_id'] for r in core.BASE['runs']))[:args.limit];records=[];start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(job,did) for did in ids]):
            records.append(f.result());print(f'{len(records)}/{len(ids)} paired DAGs; {time.time()-start:.0f}s',flush=True)
    for i,(root,name) in enumerate(((OUT,'Experiment 2a'),(OTHER,'Experiment 3-a'))):aggregate([r[i] for r in records],root,name)
    print('Both matched runs complete.',flush=True)

if __name__=='__main__':main()
