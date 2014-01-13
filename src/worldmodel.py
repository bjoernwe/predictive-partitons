import collections
import numpy as np
import random

import worldmodel_methods
import weakref


Partitioning = collections.namedtuple('Partitioning', ['labels', 'transitions', 'tree'])

SplitParams = collections.namedtuple('SplitParams', ['node',
                                                     'action',
                                                     'gain',
                                                     'test_params',
                                                     'ref_test_dict'])


class Worldmodel(object):


    def __init__(self, method='naive', seed=None):
        
        # data storage
        self._data = None        # global data storage
        self._actions = []       # a sequence of actions
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
    
    
    def get_known_actions(self):
        return set(self._action_set)


    def classify(self, data, action):
        """
        Returns the state(s) that the data belongs to according to the current 
        model. Since each action has its own model for classification, the
        action has to be specified.
        """
        return np.array(self._partitionings[action].tree._classify(data), dtype=int)
    
    
    def add_data(self, data, actions=None):
        """
        Adds a matrix of new observations to the node. The data is interpreted 
        as one observation following the previous one. This is important to 
        calculate the transition probabilities.
        """

        # check for dimensionality of x
        data = np.atleast_2d(data)

        # data length
        n = self.get_number_of_samples()
        m = data.shape[0]
        N = n + m
        
        # prepare list of actions
        if actions is None:
            actions = [None for _ in range(m-1)]
            
        # add missing action
        if self._data is not None and len(actions) == m-1:
            actions = [None] + actions
        
        # make sure right number of actions
        if self._data is not None:
            assert len(actions) == m
        else:
            assert len(actions) == m-1
            
        # update set of actions
        self._action_set = self._action_set.union(set(actions))
        
        # initialize/update partition structures
        for action in self._action_set:

            # this case happens if an action was not observed before
            # thus an empty tree is created, an empty transition matrix and
            # all-zero labels if needed
            
            # initialize new structure
            if action not in self._partitionings.keys():
                
                labels = np.zeros(n, dtype=int)
                tree = self._tree_class(model=self)
                transitions = {}
                for action_2 in self._action_set:
                    transitions[action_2] = np.ones((1, 1), dtype=int) * self._actions.count(action_2)
                self._partitionings[action] = Partitioning(labels=labels, transitions=transitions, tree=tree)
                
            # update existing structure
            else:
                
                # in this case, the action is already known and thus a 
                # partitioning already exists. however, it may happen that the
                # new data brings new actions that also need transition matrices
                # (yet empty).
                
                partitioning = self._partitionings[action]
                K = partitioning.tree.get_number_of_leaves()
                for action_2 in self._action_set:
                    if action_2 not in partitioning.transitions.keys():
                        partitioning.transitions[action_2] = np.zeros((K, K), dtype=int)
                        
        # store data in model
        if self._data is None:
            first_data = 0
            first_source = 0
            self._data = data
            self._actions = actions
        else:
            first_data = n
            first_source = first_data - 1
            self._data = np.vstack([self._data, data])
            self._actions = self._actions + actions
            
        # same number of actions and data points?
        assert self._data.shape[0] == len(self._actions) + 1
        
        # calculate new labels, and append
        for action in self._action_set:
            partitioning = self._partitionings[action]
            labels = self.classify(data, action=action)
            new_labels = np.hstack([partitioning.labels, labels])
            self._partitionings[action] = partitioning._replace(labels=new_labels)
            assert len(self._partitionings[action].labels) == N

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

        assert np.sum(P) == self.get_number_of_samples() - 1
        return P


    def _calc_best_split(self, active_action):
        """
        Calculates the gain for each state and returns a split-object for the
        best one
        """
        
        if self.data is None:
            return None
                
        best_split = None

        tree = self._partitionings[active_action].tree
        for leaf in tree.get_leaves():
            test_params = tree.calc_test_params(active_action=active_action)
            gain, ref_test_dict = leaf.calc_local_gain(active_action=active_action, test_params=test_params)
            if best_split is None or gain > best_split.gain:
                best_split = SplitParams(node = weakref.proxy(self),
                                         action = active_action, 
                                         gain = gain, 
                                         test_params = test_params,
                                         ref_test_dict = ref_test_dict)
        return best_split
    
    
    def split(self, action=None, min_gain=float('-inf')):
        
        if action is None:
            actions = self.get_known_actions()
        else:
            actions = [action]
            
        for a in actions:
            split_params = self._calc_best_split(active_action=a)
            if split_params is not None and split_params.gain >= min_gain:
                self._partitionings[a].tree.split(split_params=split_params)
                
        return


    def _recalculate_transition_matrices(self, action, new_labels, index):
        """
        Calculates a new transition matrix with the split index -> index & index+1.
        """
        assert self._partitionings[action].tree.get_leaf(index).is_leaf()
        
        N = len(new_labels)
        #S = np.sum(self._partitionings[action].transitions)
        
        for a in self.get_known_actions():
        
            # new transition matrix
            new_trans = np.array(self.transitions[action])
            # split current row and set to zero
            new_trans[index,:] = 0
            new_trans = np.insert(new_trans, index, 0, axis=0)  # new row
            # split current column and set to zero
            new_trans[:,index] = 0
            new_trans = np.insert(new_trans, index, 0, axis=1)  # new column
            
            # update all transitions from or to current state
            for i in range(N-1):
                source = self.labels[i]
                target = self.labels[i+1]
                if self.actions is None or self.actions[i+1] == action:
                    if source == index or target == index:
                        new_source = new_labels[i]
                        new_target = new_labels[i+1]
                        new_trans[new_source, new_target] += 1
    
            assert np.sum(self.transitions[action]) == S
            assert np.sum(new_trans) == S
            
        return new_trans
    


if __name__ == '__main__':
    pass    