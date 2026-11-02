"""Variable-size vectorized replay, retaining root-only dispatch semantics."""
import numpy as np
from agent_ifc.reproduction.support import prepare

def evaluate_batch(task,orders,deliveries,topology,sources,activation):
    """Vectorize independent executions; preserve sequential task and delivery events."""
    agents,hub,paths=prepare(task,topology)
    idx={n['id']:i for i,n in enumerate(topology['nodes'])}
    state=np.zeros((len(orders),len(idx)),dtype=bool)
    if activation=='initial':
        for source in sources:state[:,idx[agents[source]]]=True
    outputs=np.zeros((len(orders),max(agents)+1),dtype=bool)
    edge_set={tuple(e) for e in task['edges']}
    parents={t:sum(v==t for u,v in task['edges']) for t in agents}
    def route(rows,path,payload):
        payload=payload.copy()
        for a,b in zip(path,path[1:]):
            payload |= state[rows,idx[a]]
            state[rows,idx[b]] |= payload
        state[rows,idx[path[-1]]] |= payload
    for pos in range(orders.shape[1]):
        for target in agents:
            rows=np.flatnonzero(orders[:,pos]==target)
            if not len(rows):continue
            if hub and parents[target]==0:
                dispatch=rows
                route(dispatch,paths['orchestrator',agents[target]],state[dispatch,idx['orchestrator']])
            for slot in range(parents[target]):
                parent_sources=deliveries[rows,target,slot]
                for source in np.unique(parent_sources):
                    rr=rows[parent_sources==source]
                    route(rr,paths[agents[source],agents[target]],outputs[rr,source] | state[rr,idx[agents[source]]])
            if target in sources:state[rows,idx[agents[target]]]=True
            outputs[rows,target]=state[rows,idx[agents[target]]]
    return outputs[:,[n['id'] for n in task['nodes']]]

