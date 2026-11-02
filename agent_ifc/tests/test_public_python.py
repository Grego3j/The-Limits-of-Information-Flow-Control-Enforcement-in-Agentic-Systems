import random,unittest
import numpy as np
from agent_ifc.reproduction.support import OrderSampler,evaluate,build_topologies
from agent_ifc.experiments.experiment_0.engine import evaluate_batch
from agent_ifc.experiments.experiment_3a.run import source_order

class PublicPythonTests(unittest.TestCase):
    def test_completion_weights(self):
        s=OrderSampler(dict(nodes=[dict(id=t) for t in (1,2,3)],edges=[(1,3)]))
        self.assertEqual(s.total_orders,3)
        self.assertEqual(s.count(s.bits[1]),2);self.assertEqual(s.count(s.bits[2]),1)
    def test_agent_blocks_keep_ownership(self):
        owners=np.zeros(21,np.uint8);owners[1:]=[1,2,3,4,5]*4;before=owners.copy()
        perm,agents=source_order(owners,5,1)
        np.testing.assert_array_equal(before,owners);self.assertEqual(sorted(perm),list(range(1,21)))
        self.assertEqual([int(owners[t]) for t in perm],sum(([a]*4 for a in agents),[]))
    def test_packed_matches_original_boolean_replay(self):
        task=dict(nodes=[dict(id=t,agent=f'a{t}') for t in range(1,5)],edges=[(1,3),(2,3),(3,4)])
        s=OrderSampler(task);rng=random.Random(42);samples=[s.sample(rng) for _ in range(32)]
        orders=np.array([o for o,d in samples],np.int16);dd=np.full((32,5,4),-1,np.int16)
        for i,(o,d) in enumerate(samples):
            for t,parents in d.items():dd[i,t,:len(parents)]=parents
        source_sets=[{1},{2,4}];bits=np.zeros(5,np.uint32)
        for j,source in enumerate(source_sets):
            for t in source:bits[t]|=1<<j
        for top in build_topologies(5):
            packed=evaluate(task,orders,dd,top,bits)
            for j,source in enumerate(source_sets):np.testing.assert_array_equal((packed>>j)&1,evaluate_batch(task,orders,dd,top,source,'execution'))
if __name__=='__main__':unittest.main()
