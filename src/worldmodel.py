import collections
import numpy as np
import random
import weakref

from matplotlib import pyplot

import split_params
import worldmodel_methods


Partitioning = collections.namedtuple('Partitioning', ['labels', 'transitions', 'tree'])


class Worldmodel(object):


    def __init__(self, method='naive', seed=None):
        
        # data storage
        self.data = None        # global data storage
        self.actions = []       # a sequence of actions
        self._action_set = set()
        
        # partitionings for each action (including labels, transitions and tree)
        self.partitionings = {}

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
        return
            
            
    def get_input_dim(self):
        """
        Returns the input dimensionality of the model.
        """
        return self.data.shape[1]
    
    
    def get_number_of_samples(self):
        if self.data is None:
            return 0
        return self.data.shape[0]
    
    
    def get_known_actions(self):
        return set(self._action_set)


    def classify(self, data, action):
        """
        Returns the state(s) that the data belongs to according to the current 
        model. Since each action has its own model for classification, the
        action has to be specified.
        """
        return np.array(self.partitionings[action].tree.classify(data), dtype=int)
    
    
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
        if self.data is not None and len(actions) == m-1:
            actions = [None] + actions
        
        # make sure right number of actions
        if self.data is not None:
            assert len(actions) == m
        else:
            assert len(actions) == m-1
            
        # update set of actions
        self._action_set.update(actions)
        
        # initialize/update partition structures
        for action in self._action_set:

            # this case happens if an action was not observed before
            # thus an empty tree is created, an empty transition matrix and
            # all-zero labels if needed
            
            # initialize new structure
            if action not in self.partitionings.keys():
                
                labels = np.zeros(n, dtype=int)
                tree = self._tree_class(model=self)
                transitions = {}
                for action_2 in self._action_set:
                    transitions[action_2] = np.ones((1, 1), dtype=int) * self.actions.count(action_2)
                self.partitionings[action] = Partitioning(labels=labels, transitions=transitions, tree=tree)
                
            # update existing structure
            else:
                
                # in this case, the action is already known and thus a 
                # partitioning already exists. however, it may happen that the
                # new data brings new actions that also need transition matrices
                # (yet empty).
                
                partitioning = self.partitionings[action]
                K = partitioning.tree.get_number_of_leaves()
                for action_2 in self._action_set:
                    if action_2 not in partitioning.transitions.keys():
                        partitioning.transitions[action_2] = np.zeros((K, K), dtype=int)
                        
        # store data in model
        if self.data is None:
            first_data = 0
            first_source = 0
            self.data = data
            self.actions = actions
        else:
            first_data = n
            first_source = first_data - 1
            self.data = np.vstack([self.data, data])
            self.actions = self.actions + actions
            
        # same number of actions and data points?
        assert self.data.shape[0] == len(self.actions) + 1
        
        # calculate new labels, and append
        for action in self._action_set:
            partitioning = self.partitionings[action]
            labels = self.classify(data, action=action)
            new_labels = np.hstack([partitioning.labels, labels])
            self.partitionings[action] = partitioning._replace(labels=new_labels)
            assert len(self.partitionings[action].labels) == N

        # add references of new data to corresponding partitions            
        for action in self._action_set:
            partitioning = self.partitionings[action]
            leaves = partitioning.tree.get_leaves()
            for i in range(first_data, N):
                label = partitioning.labels[i]
                leaf = leaves[label]
                leaf.data_refs.add(i)
        
        # update transition matrices
        for action in self._action_set:
            partitioning = self.partitionings[action]
            for i in range(first_source, N-1):
                source = partitioning.labels[i]
                target = partitioning.labels[i+1]
                action_2 = self.actions[i]
                partitioning.transitions[action_2][source, target] += 1
            
        for action in self._action_set:
            assert np.sum(self._merge_transition_matrices(action)) == N-1
        return
    

    def _merge_transition_matrices(self, action):
        """
        Merges the transition matrices for a certain model (action).
        """
        
        partitioning = self.partitionings[action]
        
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

        tree = self.partitionings[active_action].tree
        for leaf in tree.get_leaves():
            test_params = leaf._calc_test_params(active_action=active_action)
            split = split_params.SplitParams(node = weakref.proxy(leaf),
                                             action = active_action, 
                                             test_params = test_params)
            if split is not None:
                if best_split is None or split.get_gain() >= best_split.get_gain():
                    best_split = split
                
        return best_split
    
    
    def split(self, action=None, min_gain=float('-inf')):
        
        if action is None:
            actions = self.get_known_actions()
        else:
            actions = [action]
            
        for a in actions:
            split_params = self._calc_best_split(active_action=a)
            if split_params is not None and split_params.get_gain() >= min_gain:
                print split_params._gain
                split_params.apply()
                
        return


    def plot_data(self, show_plot=True):
        """
        Plots all the data that is stored in the model in light gray.
        """
        
        pyplot.plot(self.data[:,0], self.data[:,1], '.', color='silver')
            
        if show_plot:
            pyplot.show()
            
        return


    def plot_data_colored_for_state(self, active_action, show_plot=True):
        """
        Plots all the data that is stored in the tree with color and shape
        according to the learned state.
        """
        
        # fancy shapes and colors
        symbols = ['o', '^', 'd', 's', '*']
        colormap = pyplot.cm.get_cmap('prism')
        pyplot.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.98, 7)])
        
        # data for the different classes
        all_leaves = self.partitionings[active_action].tree.get_leaves()
        for i, leaf in enumerate(all_leaves):
            data = leaf.get_data()
            pyplot.plot(data[:,0], data[:,1], symbols[i%len(symbols)])
                
        if show_plot:
            pyplot.show()
            
        return
    


if __name__ == '__main__':

    N = 100000
    np.random.seed(0)
    data = np.random.random((N, 2))
    actions = [i%2 for i in range(N-1)]
    model = Worldmodel(method='naive', seed=None)
    model.add_data(data=data, actions=actions)
    #model.split(action=None)
    for i in range(4):
        model.split(action=0)
    #for i, action in enumerate(model.get_known_actions()):
        pyplot.subplot(1, 4, i+1)
        model.plot_data_colored_for_state(active_action=0, show_plot=False)
    pyplot.show()
    