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
        self.failUnless(len(model.actions) == N-1)
        self.failUnless(np.sum(model.partitionings[0].get_merged_transition_matrices()) == N-1)
        self.failUnless(np.sum(model.partitionings[1].get_merged_transition_matrices()) == N-1)
        self.failUnless(np.count_nonzero(model.actions == -1) == 0)

        data = np.random.random((100, 2))
        model.add_data(data=data, actions=actions)
        
        self.failUnless(model.get_known_actions() == set([-1, 0, 1]))
        self.failUnless(model.get_number_of_samples() == 2*N)
        self.failUnless(len(model.actions) == 2*N-1)
        self.failUnless(np.sum(model.partitionings[-1].get_merged_transition_matrices()) == 2*N-1)
        self.failUnless(np.sum(model.partitionings[0].get_merged_transition_matrices()) == 2*N-1)
        self.failUnless(np.sum(model.partitionings[1].get_merged_transition_matrices()) == 2*N-1)
        self.failUnless(np.count_nonzero(model.actions == -1) == 1)
        
        for a in [-1, 0, 1]:
            self.failUnless(model.partitionings[a].transitions[-1] == 1)

        data = np.random.random((100, 2))
        actions = [2 for _ in range(N)]
        model.add_data(data=data, actions=actions)

        self.failUnless(model.get_known_actions() == set([-1, 0, 1, 2]))
        self.failUnless(model.get_number_of_samples() == 3*N)
        self.failUnless(len(model.actions) == 3*N-1)
        self.failUnless(np.sum(model.partitionings[-1].get_merged_transition_matrices()) == 3*N-1)
        self.failUnless(np.sum(model.partitionings[0].get_merged_transition_matrices()) == 3*N-1)
        self.failUnless(np.sum(model.partitionings[1].get_merged_transition_matrices()) == 3*N-1)
        self.failUnless(np.sum(model.partitionings[2].get_merged_transition_matrices()) == 3*N-1)
        self.failUnless(np.count_nonzero(model.actions == -1) == 1)

        for a in [-1, 0, 1]:
            self.failUnless(model.partitionings[a].transitions[-1] == 1)
            
            
    def testBasics(self):

        N = 100
        data = np.random.random((100, 2))
        actions = [i%2 for i in range(N-1)]
        model = worldmodel.Worldmodel(method='naive', seed=None)
        model.add_data(data=data, actions=actions)
        model.split(action=None)



if __name__ == "__main__":
    unittest.main()
    