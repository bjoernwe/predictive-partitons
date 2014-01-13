import numpy as np
import unittest

import worldmodel


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testAddData(self):
        
        N = 100
        data = np.random.random((100, 2))
        actions = [i%2 for i in range(N-1)]
        model = worldmodel.Worldmodel(method='naive', seed=None)
        model.add_data(data=data, actions=actions)

        self.failUnless(model.get_known_actions() == set([0, 1]))
        self.failUnless(model.get_number_of_samples() == N)
        self.failUnless(len(model._actions) == N-1)
        self.failUnless(np.sum(model._merge_transition_matrices(action=0)) == N-1)
        self.failUnless(np.sum(model._merge_transition_matrices(action=1)) == N-1)

        data = np.random.random((100, 2))
        model.add_data(data=data, actions=actions)
        
        self.failUnless(model.get_known_actions() == set([0, 1, None]))
        self.failUnless(model.get_number_of_samples() == 2*N)
        self.failUnless(len(model._actions) == 2*N-1)
        self.failUnless(np.sum(model._merge_transition_matrices(action=None)) == 2*N-1)
        self.failUnless(np.sum(model._merge_transition_matrices(action=0)) == 2*N-1)
        self.failUnless(np.sum(model._merge_transition_matrices(action=1)) == 2*N-1)
        
        for a in [0, 1, None]:
            self.failUnless(model._partitionings[a].transitions[None] == 1)

        data = np.random.random((100, 2))
        actions = [2 for _ in range(N)]
        model.add_data(data=data, actions=actions)

        self.failUnless(model.get_known_actions() == set([0, 1, 2, None]))
        self.failUnless(model.get_number_of_samples() == 3*N)
        self.failUnless(len(model._actions) == 3*N-1)
        self.failUnless(np.sum(model._merge_transition_matrices(action=None)) == 3*N-1)
        self.failUnless(np.sum(model._merge_transition_matrices(action=0)) == 3*N-1)
        self.failUnless(np.sum(model._merge_transition_matrices(action=1)) == 3*N-1)
        self.failUnless(np.sum(model._merge_transition_matrices(action=2)) == 3*N-1)

        for a in [0, 1, None]:
            self.failUnless(model._partitionings[a].transitions[None] == 1)
            
            
    def testBasics(self):

        N = 100
        data = np.random.random((100, 2))
        actions = [i%2 for i in range(N-1)]
        model = worldmodel.Worldmodel(method='naive', seed=None)
        model.add_data(data=data, actions=actions)
        model.split(action=None)



if __name__ == "__main__":
    unittest.main()
    