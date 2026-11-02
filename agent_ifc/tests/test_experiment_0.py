import random
import unittest
import numpy as np
from agent_ifc.experiments.experiment_0.run import execution_samples
from agent_ifc.experiments.experiment_0.engine import evaluate_batch


class ExperimentZeroTests(unittest.TestCase):
    def test_exhaustive_joint_orders(self):
        # Two task orders times two delivery permutations, including a zero ID.
        task=dict(nodes=[dict(id=i) for i in range(3)],edges=[(0,2),(1,2)])
        orders,deliveries,info=execution_samples(task,4,random.Random(42))
        self.assertEqual(info['mode'],'exhaustive')
        self.assertEqual(info['joint_order_count'],'4')
        pairs={(tuple(o),tuple(d[2,:2])) for o,d in zip(orders,deliveries)}
        self.assertEqual(len(pairs),4)
        self.assertEqual({p[0] for p in pairs},{(0,1,2),(1,0,2)})
        self.assertEqual({p[1] for p in pairs},{(0,1),(1,0)})
        sampled=execution_samples(task,3,random.Random(42))
        self.assertEqual(sampled[2]['mode'],'sampled')
        self.assertEqual(len(sampled[0]),3)
        self.assertTrue(all((tuple(o),tuple(d[2,:2])) in pairs for o,d in zip(sampled[0],sampled[1])))

    def test_delayed_source_and_persistent_ownership(self):
        task=dict(nodes=[dict(id=0,agent='a1'),dict(id=3,agent='a1'),dict(id=8,agent='a2')],edges=[])
        orders=np.array([[0,3,8],[3,0,8]])
        deliveries=np.full((2,9,3),-1)
        ideal=dict(id='ideal',nodes=[dict(id='a1'),dict(id='a2')],edges=[])
        labels=evaluate_batch(task,orders,deliveries,ideal,{3},'execution')
        self.assertEqual(labels.tolist(),[[False,True,False],[True,True,False]])


if __name__=='__main__':unittest.main()
