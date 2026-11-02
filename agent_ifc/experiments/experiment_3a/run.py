"""Original paired-run helpers with a Python simulator replacing the native kernel."""
import hashlib,io,json,random,zipfile
from pathlib import Path
import numpy as np
from agent_ifc.reproduction.support import paired_inputs, python_kernel
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
AGENTS=list(range(5,21)); BUDGETS=list(range(4,21))
BASE=paired_inputs()['base']
TOPS={int(k):v for k,v in BASE['topologies'].items()}
KEYS=BASE['topology_order']

def atomic_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def source_order(owners,k,did):
    rng=random.Random(20261001+did*100+k)
    agents=list(range(1,k+1));rng.shuffle(agents)
    tools=[]
    for a in agents:
        group=[t for t in range(1,21) if owners[t]==a]
        rng.shuffle(group);tools.extend(group)
    assert sorted(tools)==list(range(1,21))
    return tools,agents

def digest(x):return hashlib.sha256(x).hexdigest()

def save_samples(path, **arrays):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for key,value in arrays.items():
            buffer=io.BytesIO();np.save(buffer,value,allow_pickle=False)
            z.writestr(key+'.npy',buffer.getvalue())

def kernel(k,orders,deliveries,parents,owners,sources):
    return python_kernel(k,orders,deliveries,parents,owners,sources,TOPS[k])

if __name__=='__main__':
    raise SystemExit('Run python -m agent_ifc.experiments.experiment_2a.run to launch both 2a and 3a.')
