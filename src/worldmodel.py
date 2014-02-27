import numpy as np
import random

from matplotlib import pyplot

from partitioning import Partitioning
import split_params
import worldmodel_methods



class Worldmodel(object):


    def __init__(self, method='naive', uncertainty_prior=10, factorization_weight=0.9, seed=None):
        
        # data storage
        self.data = None                        # global data storage
        self.actions = np.empty(0, dtype=int)   # an array of actions
        self.uncertainty_prior = uncertainty_prior
        self.factorization_weight = factorization_weight
        self.partitionings = {}
        self._action_set = set()

        #assert gain_measure in ['local', 'global']
        #self.gain_measure = gain_measure
        
        # root node of tree
        assert method in ['naive', 'fast', 'predictive']
        self._method = method
        if method == 'naive':
            self._tree_class = worldmodel_methods.WorldmodelTrivial
        elif method == 'fast':
            self._tree_class = worldmodel_methods.WorldmodelFast
        elif method == 'predictive':
            self._tree_class = worldmodel_methods.WorldmodelGPFA
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
    
    
    def get_partitioning(self, action):
        return self.partitionings[action]


    def classify(self, data, action):
        """
        Returns the state(s) that the data belongs to according to the current 
        model. Since each action has its own model for classification, the
        action has to be specified.
        """
        return self.partitionings[action].classify(data)
    

    def add_data(self, data, actions=None):
        """
        Adds a matrix of new observations to the node. The data is interpreted 
        as one observation following the previous one. This is important to 
        calculate the transition probabilities.
        
        Actions are stored as an array of positive integer values. If an action
        is unknown (e.g., the action between two chunks of data) it's set to -1.
        """

        # check for dimensionality of x
        data = np.atleast_2d(data)

        # data length
        n = self.get_number_of_samples()
        m = data.shape[0]
        N = n + m
        
        # prepare list of actions
        if actions is None:
            actions = -np.ones(m-1, dtype=int)
            
        # add missing action
        if self.data is not None and len(actions) == m-1:
            actions = np.hstack([-1, actions])
        
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
                
                self.partitionings[action] = Partitioning(model=self, active_action=action)
                
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
            self.actions = np.array(actions, dtype=np.int)
        else:
            first_data = n
            first_source = first_data - 1
            self.data = np.vstack([self.data, data])
            self.actions = np.hstack([self.actions, actions])
            
        # same number of actions and data points?
        assert self.data.shape[0] == len(self.actions) + 1
        
        # calculate new labels, and append
        for action in self._action_set:
            partitioning = self.partitionings[action]
            new_labels = self.classify(data, action=action)
            self.partitionings[action].labels = np.hstack([partitioning.labels, new_labels])
            assert len(self.partitionings[action].labels) == N

        # add references of new data to corresponding partitions            
        for action in self._action_set:
            partitioning = self.partitionings[action]
            leaves = partitioning.tree.get_leaves()
            for leaf in leaves:
                leaf_index = leaf.get_leaf_index()
                new_refs_mask = (new_labels[first_data: N] == leaf_index)
                new_refs = np.where(new_refs_mask)[0] + first_data
                leaf.data_refs = np.hstack([leaf.data_refs, new_refs])
            
        # update transition matrices
        # TODO: could be faster
        for action in self._action_set:
            partitioning = self.partitionings[action]
            for i in range(first_source, N-1):
                source = partitioning.labels[i]
                target = partitioning.labels[i+1]
                action_2 = self.actions[i]
                partitioning.transitions[action_2][source, target] += 1
            
        for action in self._action_set:
            assert np.sum(self.partitionings[action].get_merged_transition_matrices()) == N-1
        return
    
    
    def get_data_for_refs(self, refs):
        return self.data[refs]
    
    
    def get_data_refs_for_action(self, action):
        return np.where(self.actions == action)
    
    
    def split(self, action=None, min_gain=float('-inf')):
        
        if action is None:
            actions = self.get_known_actions()
        else:
            actions = [action]
            
        for a in actions:
            split_params = self.partitionings[a].calc_best_split()
            if split_params is not None and split_params.get_gain() >= min_gain:
                print split_params.get_gain()
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
        self.partitionings[active_action].plot_data_colored_for_state(show_plot=show_plot)
        return


    def plot_state_borders(self, active_action, show_plot=True, range_x=None, range_y=None, resolution=100):
        self.partitionings[active_action].plot_state_borders(show_plot=show_plot, range_x=range_x, range_y=range_y, resolution=resolution)
        
        
    def plot_transitions(self, active_action):
        self.partitionings[active_action].plot_transitions()
    


if __name__ == '__main__':

    N = 100000
    #np.random.seed(0)
    data = np.random.random((N, 2))
    actions = [i%2 for i in range(N-1)]
    model = Worldmodel(method='fast', uncertainty_prior=100, seed=None)
    model.add_data(data=data, actions=actions)
    model.split(action=None)
    for i in range(8):
        model.split(action=0)
        pyplot.subplot(2, 4, i+1)
        model.plot_data_colored_for_state(active_action=0, show_plot=False)
    pyplot.show()
    