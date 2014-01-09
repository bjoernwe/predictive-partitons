import collections
import numpy as np
import random

import worldmodel_methods


Partitioning = collections.namedtuple('Partitioning', ['labels', 'transitions', 'tree'])


class Worldmodel(object):


    def __init__(self, method='naive', seed=None):
        
        # data storage
        self._data = None        # global data storage
        self._actions = None     # a sequence of actions
        self._action_set = set()
        
        # partitionings for each action (including labels, transitions and tree)
        self._partitionings = {}

        # root node of tree
        assert method in ['naive']
        self._method = method
        if method == 'naive':
            self._tree_class = worldmodel_methods.WorldmodelTrivial
        else:
            assert False
        
        # random generator
        self._random = random.Random()
        if seed is not None:
            self.random.seed(seed)
            
            
    def get_number_of_samples(self):
        if self._data is None:
            return 0
        return self._data.shape[0]


    def classify(self, data, action):
        """
        Returns the state(s) that the data belongs to according to the current 
        model. Since each action has its own model for classification, the
        action has to be specified.
        """
        return np.array(self._partitionings[action].tree.classify(data), dtype=int)
    
    
    def add_data(self, data, actions=None):
        """
        Adds a matrix of new observations to the node. The data is interpreted 
        as one observation following the previous one. This is important to 
        calculate the transition probabilities.
        """

        # check for dimensionality of x
        data = np.atleast_2d(data)

        # data length
        n = data.shape[0]
        N = self.get_number_of_samples() + n
        
        # prepare list of actions
        if actions is None:
            actions = [None for _ in range(n-1)]
            
        # add missing action
        if self._data is not None and len(actions) == n-1:
            actions = [None] + actions
        
        # make sure right number of actions
        if self._data is not None:
            assert len(actions) == n
        else:
            assert len(actions) == n-1
            
        # store data in model
        if self._data is None:
            first_data = 0
            first_source = 0
            self._data = data
            self._actions = actions
        else:
            first_data = self.get_number_of_samples()
            first_source = first_data - 1
            self._data = np.vstack([self._data, data])
            self._actions = self._actions + actions
            
        # same number of actions and data points?
        assert self._data.shape[0] == len(self._actions)+1
        
        # remember actions
        self._action_set = self._action_set.union(set(actions))
        
        # initialize/update partition structures
        for action in self._action_set:
            
            labels = self.classify(data, action=action)
            
            # initialize new structure
            if action not in self._partitionings.keys():
                
                tree = self._tree_class(model=self)
                transitions = {}
                for action_2 in self._action_set:
                    transitions[action_2] = np.zeros((1, 1), dtype=int) 
                self._partitionings[action] = Partitioning(labels=labels, transitions=transitions, tree=tree)
                
            # update existing structure
            else:
                
                self._partitionings[action].labels = np.hstack([self._partitionings[action].labels, labels])
                K = self._partitionings[action].tree.get_number_of_leaves()
                for action_2 in self._action_set:
                    if action_2 not in self._partitionings[action].transitions.keys():
                        self._partitionings[action].transitions[action_2] = np.zeros((K, K), dtype=int)

        # add references of new data to corresponding partitions            
        for action in self._action_set:
            partitioning = self._partitionings[action]
            for i in range(first_data, N):
                label = partitioning.labels[i]
                leaf = partitioning.tree.get_leaf(label)
                leaf._dat_ref.append(i)
        
        # update transition matrices
        for action in self._action_set:
            partitioning = self._partitionings[action]
            for i in range(first_source, N-1):
                source = partitioning.labels[i]
                target = partitioning.labels[i+1]
                action_2 = self._actions[i]
                partitioning.transitions[action_2][source, target] += 1
            
        for action in self._action_set:
            assert np.sum(self._merge_transition_matrices(action)) == N-1
        return
    

    def _merge_transition_matrices(self, action):
        """
        Merges the transition matrices for a certain model (action).
        """
        
        partitioning = self._partitionings[action]
        
        K = partitioning.tree.get_number_of_leaves()
        P = np.zeros((K, K), dtype=int)
        
        for a in self._action_set:
            P += partitioning.transitions[a]

        assert np.sum(P) == self._data.shape[0] - 1
        return P
    


if __name__ == '__main__':
    
    model = Worldmodel()
    model.add_data(data=np.random.random((10, 2)), actions=None)
    