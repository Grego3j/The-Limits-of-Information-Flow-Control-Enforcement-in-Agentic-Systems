"""Small Python replacements for deleted shared modules; see PACKAGING_CHANGES.md."""
from collections import deque
from functools import lru_cache
import gzip,json,math,random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
@lru_cache(None)
def paired_inputs():
    with gzip.open(ROOT/'reproduction/paired_inputs.json.gz','rt') as f:return json.load(f)

def build_topologies(k,seed=42):
    if seed!=42:raise ValueError('Release includes the original seed-42 catalogs only')
    return json.loads((ROOT/f'topologies/agents_{k:02d}_topologies.json').read_text())['topologies']
COLORS=[n['color'] for n in build_topologies(20)[0]['nodes']]

class OrderSampler:
    def __init__(self,task):
        self.ids=sorted(n['id'] for n in task['nodes']);idx={t:i for i,t in enumerate(self.ids)}
        self.parent_ids={t:sorted(u for u,v in task['edges'] if v==t) for t in self.ids}
        self.masks={t:sum(1<<idx[u] for u in self.parent_ids[t]) for t in self.ids};self.bits={t:1<<idx[t] for t in self.ids};self.full=(1<<len(self.ids))-1
        self.count=lru_cache(None)(self._count);self.total_orders=self.count(0);self.total_deliveries=math.prod(math.factorial(len(p)) for p in self.parent_ids.values())
    def ready(self,done):return [t for t in self.ids if not done&self.bits[t] and done&self.masks[t]==self.masks[t]]
    def _count(self,done):return 1 if done==self.full else sum(self.count(done|self.bits[t]) for t in self.ready(done))
    def sample(self,rng):
        order=[];done=0
        while done!=self.full:
            draw=rng.randrange(self.count(done))
            for t in self.ready(done):
                w=self.count(done|self.bits[t])
                if draw<w:break
                draw-=w
            order.append(t);done|=self.bits[t]
        dd={t:rng.sample(p,len(p)) for t,p in self.parent_ids.items()}
        return order,dd

def prepare(task,top):
    agents={n['id']:n['agent'] for n in task['nodes']};adj={n['id']:set() for n in top['nodes']}
    for a,b in top['edges']:adj[a].add(b);adj[b].add(a)
    paths={}
    for a in adj:
        for b in adj:
            if top['id']=='ideal':paths[a,b]=[a] if a==b else [a,b];continue
            q=deque([[a]]);seen={a}
            while q:
                path=q.popleft()
                if path[-1]==b:paths[a,b]=path;break
                for v in sorted(adj[path[-1]]-seen):seen.add(v);q.append(path+[v])
    return agents,'orchestrator' in adj,paths

def realize(task,top,order,delivery):return task,top,order,delivery

def reference_labels(realized,top,sources,activation):
    task,_,order,delivery=realized;owners,hub,paths=prepare(task,top)
    state={owners[t] for t in sources} if activation=='initial' else set();outputs={}
    def route(path,payload):
        for a in path:
            payload=payload or a in state
            if payload:state.add(a)
    for t in order:
        parents=delivery[t]
        if hub and not parents:route(paths['orchestrator',owners[t]],False)
        for p in parents:route(paths[owners[p],owners[t]],outputs[p])
        if t in sources:state.add(owners[t])
        outputs[t]=int(owners[t] in state)
    return outputs

def evaluate(task,orders,deliveries,top,bits):
    """Original experiment-0 replay, vectorized over samples and source-budget bits."""
    agents,hub,paths=prepare(task,top);idx={n['id']:i for i,n in enumerate(top['nodes'])}
    state=np.zeros((len(orders),len(idx)),np.uint32);outputs=np.zeros((len(orders),max(agents)+1),np.uint32)
    parents={t:sum(v==t for u,v in task['edges']) for t in agents}
    def route(rows,path,payload):
        payload=payload.copy()
        for a,b in zip(path,path[1:]):
            payload |= state[rows,idx[a]];state[rows,idx[b]] |= payload
        state[rows,idx[path[-1]]] |= payload
    for pos in range(orders.shape[1]):
        for target in agents:
            rows=np.flatnonzero(orders[:,pos]==target)
            if not len(rows):continue
            if hub and parents[target]==0:route(rows,paths['orchestrator',agents[target]],state[rows,idx['orchestrator']])
            for slot in range(parents[target]):
                ps=deliveries[rows,target,slot]
                for source in np.unique(ps):
                    rr=rows[ps==source];route(rr,paths[agents[source],agents[target]],outputs[rr,source]|state[rr,idx[agents[source]]])
            state[rows,idx[agents[target]]] |= bits[target]
            outputs[rows,target]=state[rows,idx[agents[target]]]
    return outputs[:,[n['id'] for n in task['nodes']]]

def python_kernel(k,orders,deliveries,parents,owners,sources,topologies):
    """Drop-in Python replacement for the paired studies' native kernel interface."""
    assert np.all(owners==owners[0]),'2a/3a have fixed ownership across budgets'
    task=dict(nodes=[dict(id=t,agent=f'a{int(owners[0,t])+1}') for t in range(1,21)],edges=[(int(p),t) for t in range(1,21) for p in deliveries[0,t,:int(parents[t])]])
    bits=np.zeros(21,np.uint32)
    for bi,source in enumerate(sources):
        for t in range(1,21):
            if int(source)&(1<<t):bits[t]|=np.uint32(1<<bi)
    counts=np.zeros((len(sources),len(topologies),len(orders)),np.uint8);masks=np.zeros_like(counts,dtype=np.uint32)
    weights=np.uint32(1)<<np.arange(1,21,dtype=np.uint32)
    for j,top in enumerate(topologies):
        labels=evaluate(task,orders,deliveries,top,bits)
        for bi in range(len(sources)):
            high=(labels>>bi)&1;counts[bi,j]=high.sum(axis=1);masks[bi,j]=high@weights
    return counts,masks
