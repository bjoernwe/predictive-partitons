import numpy as np
import unittest

import worldmodel


class TestRandomData(unittest.TestCase):


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
        
        
        
class TestOrderedData1(unittest.TestCase):

    def testSplits(self):
        
        # generate data
        N = 40+1
        data_centers = ((.25, .75), (.75, .75), (.75, .25), (.25, .25))
        data = np.empty((N, 2))
        for i in range(N):
            data[i] = data_centers[i%4]
            
        # create model
        model = worldmodel.Worldmodel(method='naive', seed=None, uncertainty_prior=0)
        model.add_data(data=data, actions=None)
        for i in range(3):
            model.split()
             
        self.failUnlessAlmostEqual(model.partitionings[-1].tree._split_params._gain, 0.0)
        self.failUnlessAlmostEqual(model.partitionings[-1].tree._children[0]._split_params._gain, 0.25162921584607989)
        self.failUnlessAlmostEqual(model.partitionings[-1].tree._children[1]._split_params._gain, 0.25162921584607989)



class TestOrderedData2(unittest.TestCase):

    def testSplits(self):
        
        # generate data
        N = 1600
        data_centers = ((.25, .75), (.75, .75), (.75, .25), (.25, .25))
        data_centers_2 = ((.625, .125), (.875, .125), (.875, .375), (.625, .375))
        data = 0.01 * np.random.randn(N, 2)
        for i in range(N/4):
            if i%4 == 2:
                for j in range(4):
                    data[4*i+j] += data_centers_2[j%4]
            else:
                for j in range(4):
                    data[4*i+j] += data_centers[i%4]
            
        # create model
        model = worldmodel.Worldmodel(method='naive', seed=None, uncertainty_prior=10)
        model.add_data(data=data, actions=None)
        for i in range(5):
            model.split()

        # test
        self.failUnless(np.array_equal(model.partitionings[-1].transitions[-1],
                        np.array([[300,  99,   0,   0,   0,   0],
                                  [  0, 300,   0,   0,   0, 100],
                                  [100,   0,   0, 100,   0,   0],
                                  [  0,   0,   0,   0, 100,   0],
                                  [  0,   0, 100,   0,   0,   0],
                                  [  0,   0, 100,   0,   0, 300]])))



if __name__ == "__main__":
    unittest.main()
    