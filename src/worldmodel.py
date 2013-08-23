import collections
import math
import numpy as np
import random
import scipy.linalg
import scipy.sparse.linalg
import scipy.spatial.distance
import sklearn.manifold
import traceback

from matplotlib import pyplot

import mdp


Stats = collections.namedtuple('Stats', ['n_states',
                                         'n_nodes', 
                                         'norm', 
                                         'entropy', 
                                         'entropy_normalized', 
                                         'mutual_information'])

SplitResult = collections.namedtuple('SplitResult', ['node',
                                                     'action',
                                                     'gain',
                                                     'classifier'])

class WorldModel(object):

    def __init__(self, method='spectral', seed=None):
        
        # data storage
        self.data = None        # global data storage
        #self.transitions = None
        self.actions = None     # either None or a list of actions
        self.stats = []
        self._min_class_size = 100
        
        self.transitions = {}
        self.transitions[None] = np.array([[0]])
        
        # root node of tree
        assert method in ['factorize' ,'pca', 'spectral']
        self.method = method
        if method == 'factorize':
            self.tree = WorldModelPCA(model=self)
        elif method == 'pca':
            self.tree = WorldModelPCA(model=self)
        elif method == 'spectral':
            self.tree = WorldModelSpectral(model=self)
        else:
            print 'Should not happen!'
        
        # random generator
        self.random = random.Random()
        if seed is not None:
            self.random.seed(seed)


    def add_data(self, data, actions=None):
        """
        Adds a matrix x of new observations to the node. The data is
        interpreted as one observation following the previous one. This is
        important to calculate the transition probabilities.
        
        The actions are a interpreted as actions that *preceded* each step. If
        you don't know the action that preceded the first data point, it's okay
        to leave the list of actions shorter by one. The missing action will be 
        filled with 'None' and the transaction ignored during most calculations.
        
        If there has been data before, the new data is appended.
        """

        # check for dimensionality of x
        data = np.atleast_2d(data)

        # initialize actions
        n = data.shape[0]
        #if actions is None:
        #    actions = [None for _ in range(n)]
        if actions and len(actions) < n:
            actions = [None] + actions
        assert actions is None or len(actions) == n

        # calculate labels for new data
        labels = np.empty(n, dtype=int)
        for i in range(n):
            labels[i] = self.classify(data[i])
            
        # store data in model
        if self.data is None:
            first_data = 0
            first_source = 0
            self.data = data
            self.labels = labels
            self.actions = actions
        else:
            first_data = self.data.shape[0]
            first_source = first_data - 1
            self.data = np.vstack([self.data, data])
            self.labels = np.hstack([self.labels, labels])
            if self.actions is not None:
                self.actions = self.actions + actions
            
        # same number of actions and data points
        assert actions is None or self.data.shape[0] == len(self.actions)
        assert self.data.shape[0] == self.labels.shape[0]

        if self.actions is None:
            
            # add references to data in all leaves
            all_leaves = self.tree.get_leaves()
            N = self.data.shape[0]
            for i in range(first_data, N):
                state = self.labels[i]
                leaf  = all_leaves[state]
                leaf.dat_ref.append(i)
                # reset cache because of new data
                leaf.split_cache = {}
                    
            # create global transition matrices (for each action)
            K = len(all_leaves)
            if self.transitions is None:
                self.transitions = {}
            if None not in self.transitions.keys():
                self.transitions[None] = np.zeros((K,K))
                
            # update transition matrices
            for i in range(first_source, N-1):
                source = self.labels[i]
                target = self.labels[i+1]
                self.transitions[None][source, target] += 1
            
        else:

            # add references to data in all leaves
            all_leaves = self.tree.get_leaves()
            N = self.data.shape[0]
            for i in range(first_data, N):
                state = self.labels[i]
                leaf  = all_leaves[state]
                leaf.dat_ref.append(i)
                # TODO don't delete splits for all actions!
                leaf.split_cache = {}  # reset cache because of new data
                    
            # create global transition matrices (for each action)
            K = len(all_leaves)
            if self.transitions is None:
                self.transitions = {}
            action_set = set(actions).union([None])
            for action in action_set:
                if action not in self.transitions.keys():
                    self.transitions[action] = np.zeros((K,K))
                
            # update transition matrices
            for i in range(first_source, N-1):
                source = self.labels[i]
                target = self.labels[i+1]
                action = self.actions[i+1]
                self.transitions[action][source, target] += 1
                # reset cache for that action
                if source == target:
                    state = self.labels[i]
                    leaf  = all_leaves[state]
                    #leaf.split_cache = {}
                    if leaf.split_cache.has_key(action):
                        leaf.split_cache.pop(action)
                        leaf.split_cache.pop(None)
                        leaf.split_cache[None] = max(leaf.split_cache.values())
            
        assert np.sum(self._merge_transition_matrices(transitions=self.transitions)) == N-1
        return


    def learn(self, action=None, min_gain=0.02):
        """
        Learns the model.
        """
        
        # init stats
        if len(self.stats) == 0:
            self.stats.append(self._calc_stats(transitions=self.transitions))

        # split as long as it's interesting
        gain = 0
        while True:
            split = self.single_splitting_step(action=action, min_gain=min_gain)
            if split is None or split.gain < min_gain:
                break
            else:
                gain = split.gain
                print 'split with gain', gain, '\n'
            
        return gain
    
    
    def single_splitting_step(self, action=None, min_gain=float('-inf')):
        """
        Calculates the gain for each state and splits the best one. Can be
        restricted to a given action.
        
        TODO: only re-calculate states with some change
        """
        if self.data is None:
            return None
                
        best_gain = float('-inf')
        best_split = None
        
        for leaf in self.tree.get_leaves():
            print 'testing leaf', leaf.get_leaf_index(), '...'
            split = leaf._calculate_best_split(action=action)
            if split is not None and leaf._reached_min_sample_size(action=action):
                print 'best split: leaf', leaf.get_leaf_index(), 'with action', split.action, 'with gain', split.gain
                if split.gain > best_gain:
                    best_gain = split.gain
                    best_split = split
                
        if best_split is not None and best_gain >= min_gain:
            print 'decided for leaf', best_split.node.get_leaf_index(), 'with action', best_split.action, 'and gain', best_split.gain
            best_split.node._apply_split(split_result=best_split)
            self.stats.append(self._calc_stats(transitions=self.transitions))
            
        return best_split
    
    
    def classify(self, x):
        """
        Returns the state that x belongs to according to the current model. If
        x is a matrix, a list is returned containing a integer state for every
        row.
        """
        return self.tree.classify(x)
                

    def get_possible_actions(self, ignore_none=False):
        """
        Returns a set of actions used so far. In principle other actions can
        be added afterwards with new data.
        """
        actions = set(self.transitions.keys())
        if ignore_none and self.actions is not None and None in actions:
            actions.remove(None)
        return actions


    def get_number_of_states(self):
        """Returns the number of states in the model."""
        return len(self.tree.get_leaves())
    
    
    def get_number_of_samples(self):
        """Returns the number of data points stored in the model."""
        return self.data.shape[0]
        
        
    def get_input_dim(self):
        """
        Returns the input dimensionality of the model.
        """
        return self.data.shape[1]
    
    
    def _split_transition_matrices(self, new_labels, index1, index2=None):
        """
        Calculates a split transition matrix for each action. The result is
        a dictionary with actions as keys.
        """
        
        N = len(new_labels)
        result = {}

        # check
        transitions = self._merge_transition_matrices()
        assert np.sum(transitions) == N-1
        
        for action in self.transitions.keys():
            result[action] = self._split_transition_matrix(action=action, new_labels=new_labels, index1=index1, index2=index2)
            
        return result


    def _split_transition_matrix(self, action, new_labels, index1, index2=None):
        """
        Calculates a new transition matrix with the split index1 -> index1 & index1+1.
        
        In special cases it might be necessary to have a split index1 -> index1 & index2.
        """
        N = len(new_labels)
        S = np.sum(self.transitions[action])
        
        assert self.tree.get_leaves()[index1].status == 'leaf'
        assert max(new_labels) == self.transitions[action].shape[0]
        
        if index2 is None:
            index2 = index1
            assert self.tree.get_leaves()[index1].status == 'leaf'

        # new transition matrix
        new_trans = np.array(self.transitions[action])
        # split current row and set to zero
        new_trans[index1,:] = 0
        new_trans = np.insert(new_trans, index2, 0, axis=0)  # new row
        # split current column and set to zero
        new_trans[:,index1] = 0
        new_trans = np.insert(new_trans, index2, 0, axis=1)  # new column
        
        # update all transitions from or to current state
        for i in range(N-1):
            source = self.labels[i]
            target = self.labels[i+1]
            if self.actions is None or self.actions[i+1] == action:
                if source == index1 or target == index1:
                    new_source = new_labels[i]
                    new_target = new_labels[i+1]
                    new_trans[new_source, new_target] += 1

        assert np.sum(self.transitions[action]) == S
        assert np.sum(new_trans) == S        
        return new_trans
    
    
    def _calc_stats(self, transitions):
        """
        Calculates statistics for a given transition matrix.
        """

        n_nodes = len(self.tree._get_nodes())
        
        P = self._merge_transition_matrices(transitions)
        K = P.shape[0]
        entropy = self._matrix_entropy(transitions=P)
        entropy_normalized = self._matrix_entropy(transitions=P, normalize=True)
        
        # norm of Q
        weights = np.sum(P, axis=1, dtype=np.float)
        probs = P / weights[:,np.newaxis]
        mu = weights / np.sum(weights)
        norm = np.sum( ( probs**2 * mu[:,np.newaxis] ) / mu[np.newaxis,:] )
        
        # mutual information
        entropy_mu = self._entropy(dist=mu)
        mutual_information = entropy_mu - entropy

        stats = Stats(n_states = K,
                      n_nodes = n_nodes, 
                      entropy = entropy, 
                      entropy_normalized = entropy_normalized, 
                      norm = norm, 
                      mutual_information = mutual_information)
        return stats
    
    
    def _merge_transition_matrices(self, transitions=None):
        """
        Merges the transition matrices of each action to a single one.
        """

		# TODO we do not want the None-transitions, do we?
        
        if transitions is None:
            transitions = self.transitions
        if transitions is None:
            return None
        K = transitions.itervalues().next().shape[0]
        
        P = np.zeros((K,K), dtype=np.int)
        for action in self.transitions.keys():
            P += self.transitions[action]

        assert np.sum(P) == len(self.labels)-1
        return P
    
    
    @classmethod
    def _matrix_entropy(cls, transitions, normalize=False):
        """
        Calculates the entropy for a given transition matrix, i.e. the 
        transition entropy of each state weighted by how often it occurs.
        """        
        K = transitions.shape[0]
        row_entropies = np.zeros(K)

        for i in range(K):
            row = transitions[i]
            row_entropies[i] = cls._entropy(dist=row, normalize=normalize, ignore_empty_classes=False)

        # weighted average
        weights = np.sum(transitions, axis=1, dtype=np.float32)
        weights /= np.sum(weights)
        entropy = np.sum(weights * row_entropies)
        return entropy
    
    
    @classmethod
    def _entropy(cls, dist, normalize=False, ignore_empty_classes=False):
        """
        Calculates the (normalized) entropy over a given probability 
        distribution.
        """

        # negative values?
        assert True not in list(dist < -0.)

        # useful variables
        trans_sum = np.sum(dist)
        K = len(dist)

        # only one class?
        if K <= 1:
            return 1.0

        # empty class?
        if trans_sum == 0:
            if not ignore_empty_classes:
                assert trans_sum != 0
            if normalize:
                return 1.0
            else:
                return np.log2(K)

        # the actual calculation
        probs = np.array(dist, dtype=np.float32) / trans_sum
        log_probs = np.zeros_like(probs)
        log_probs[probs > 0.] = np.log2( probs[probs > 0.] )
        entropy = -np.sum(probs * log_probs)

        # normalization?
        assert(entropy <= np.log2(K) + 1e-8)
        if normalize:
            entropy /= np.log2(K)

        assert(entropy >= 0)
        return entropy


    @classmethod
    def _mutual_information(cls, transition_matrix):
        """
        Calculates the mutual information between t and t+1 for a model given
        as transition matrix. 
        """
        if transition_matrix is None:
            return None
        P = transition_matrix
        assert np.sum(P) > 0
        weights = np.sum(P, axis=1, dtype=np.float32)
        mu = weights / np.sum(weights)
        entropy_mu = cls._entropy(dist=mu)
        entropy = cls._matrix_entropy(transitions=transition_matrix)
        mutual_information = entropy_mu - entropy
        return mutual_information
    
    
    @classmethod
    def _mutual_information_average(cls, transition_matrices):
        """
        Calculates the weighted average of mutual information for several models 
        given as a list of transition matrices.
        """
        weights = [np.sum(P) for P in transition_matrices]
        list_mi = [weights[i] * cls._mutual_information(P) for i, P in enumerate(transition_matrices) if weights[i] > 0]
        return np.sum(list_mi) / np.sum(weights)
            
            
    def get_transitions(self, copy=False):
        """
        Returns the (copied) dictionary of transition matrices. 
        """
        if not copy:
            return self.transitions
        else:
            trans = {}
            for a, T in self.transitions.iteritems():
                trans[a] = np.array(T, copy=True)
            return trans
        

    def get_transition_probabilities(self, action=None, soft=False):
        """
        Returns a dictionary containing for every action a matrix with 
        transition probabilities. If an action is specified only the 
        corresponding transition matrix is returned.
        """
        if action not in self.transitions.keys():
            return None
         
        if self.actions is not None and action is None:
            probs = {}
            for action in self.transitions.keys():
                probs[action] = np.array(self.transitions[action], dtype=np.float32)
                if soft:
                    probs[action] += 1
                probs[action] /= probs[action].sum(axis=1)[:, np.newaxis] # normalize
                 
        else:
            probs = np.array(self.transitions[action], dtype=np.float32) 
            if soft:
                probs[action] += 1
            probs /= probs.sum(axis=1)[:, np.newaxis]
         
        return probs

    
    def get_graph_cost_matrix(self, soft=False):
        """
        Returns a distance matrix for the learned states that may be useful for
        calculating a shortest path through the state space.
         
        The distances are calculated like this: first, we are considering the 
        probability of moving from one state to another after selecting the best 
        action for that transition. For that probability we then calculate the 
        (negative) logarithm. The Dijkstra algorithm for instance works by 
        summing up distances which would results in the (log-) product of our 
        probability values.
        """
         
        # helper variables
        if self.get_number_of_samples() <= 1:
            return np.ones((1,1))
        probs = self.get_transition_probabilities(soft=soft)
        actions = probs.keys()
        if self.actions is not None and None in actions:
            actions.remove(None)
        num_actions = len(actions)
        num_states = self.get_number_of_states()
         
        # tensor that holds all transition matrices
        T = np.zeros((num_actions, num_states, num_states))
        for i, action in enumerate(actions):
            T[i] = probs[action]
             
        # from transition probabilities to affinity
        A = T.max(axis=0)
        A = -np.log(A)
        return A


    def get_state_means(self):
        """
        Returns a matrix with mean values for every state.
        """
        K = self.get_number_of_states()
        D = self.get_input_dim()
        means = np.zeros((K, D))
         
        for i, state in enumerate(self.tree.get_leaves()):
            means[i] = state.get_data().mean(axis=0)
             
        return means

    
    def plot_data(self, color='state', vmin=None, vmax=None, ndim=None, show_plot=True):
        """
        Plots all the data that is stored in the tree with color and shape
        according to the learned state.
        """
        
        symbols = ['o', '^', 'd', 's', '*']
        
        if color == 'state':

            # list of data for the different classes
            data_list = []
            all_leaves = self.tree.get_leaves()
            for leaf in all_leaves:
                data = leaf.get_data()
                if data is not None:
                    if ndim is not None and ndim < self.get_input_dim():
                        data = sklearn.manifold.Isomap(n_neighbors=10, n_components=ndim).fit_transform(data)
                    data_list.append(data)
    
            # plot
            colormap = pyplot.cm.get_cmap('prism')
            pyplot.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.98, 7)])
            for i, data in enumerate(data_list):
                pyplot.plot(data[:,0], data[:,1], symbols[i%len(symbols)])
                
        elif color == 'last_gain':
            
            leaves = self.tree.get_leaves()
            if vmin is None:
                vmin = min([leaf.get_gain() for leaf in leaves])
            if vmax is None:
                vmax = max([leaf.get_gain() for leaf in leaves])
            colormap = pyplot.cm.get_cmap('summer')
            
            for leaf in leaves:
                data = leaf.get_data()
                if ndim is not None and ndim < self.get_input_dim():
                    data = sklearn.manifold.Isomap(n_neighbors=10, n_components=ndim).fit_transform(data)
                gain = leaf.get_gain()
                colors = [gain for _ in range(data.shape[0])]
                pyplot.scatter(x=data[:,0], y=data[:,1], c=colors, cmap=colormap, edgecolors='none', vmin=vmin, vmax=vmax)
                
        else:
            
            data = self.data
            if ndim is not None and ndim < self.get_input_dim():
                data = np.array(data)
                data = sklearn.manifold.Isomap(n_neighbors=10, n_components=ndim).fit_transform(data)
            pyplot.plot(data[:,0], data[:,1], '.')
            
        if show_plot:
            pyplot.show()
        return
    
    
    def plot_states(self, show_plot=True, range_x=None, range_y=None, resolution=100):
        """
        Shows a contour plot of the learned states (2D). 
        """
        
        data = self.data
        if data.shape[1] > 2:
            data = np.array(self.data)
            data = sklearn.manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(data)
        K = len(self.tree.get_leaves())
        
        if range_x is None:
            range_x = [np.min(data[:,0]), np.max(data[:,0])]
            
        if range_y is None:
            range_y = [np.min(data[:,1]), np.max(data[:,1])]
            
        x = np.linspace(range_x[0], range_x[1], resolution)
        y = np.linspace(range_y[0], range_y[1], resolution)
        X, Y = np.meshgrid(x, y)
        v_classify = np.vectorize(lambda x, y: self.classify(np.array([x,y])))
        Z = v_classify(X, Y)
        pyplot.contourf(X, Y, Z, levels = range(-1, K))
        
        if show_plot:
            pyplot.show()
        return


    def plot_state_borders(self, show_plot=True, range_x=None, range_y=None, resolution=100):
        """
        Shows a contour plot of the learned state borders (2D). 
        """
        
        data = self.data
        if data.shape[1] > 2:
            data = np.array(self.data)
            data = sklearn.manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(data)
        K = len(self.tree.get_leaves())
        
        if range_x is None:
            range_x = [np.min(data[:,0]), np.max(data[:,0])]
            
        if range_y is None:
            range_y = [np.min(data[:,1]), np.max(data[:,1])]
            
        x = np.linspace(range_x[0], range_x[1], resolution)
        y = np.linspace(range_y[0], range_y[1], resolution)
        X, Y = np.meshgrid(x, y)
        v_classify = np.vectorize(lambda x, y: self.classify(np.array([x,y])))
        Z = v_classify(X, Y)
        pyplot.contour(X, Y, Z, levels = range(-1, K), colors='b', linewidths=1)
        
        if show_plot:
            pyplot.show()
        return
         

    def plot_stats(self, show_plot=True):
        """
        Plots how the models benchmark values have developed during training.
        """
        stats = np.vstack(self.stats)
        pyplot.plot(stats)
        pyplot.legend(list(self.stats[0]._fields)[0:], loc=2)
         
        if show_plot:
            pyplot.show()
        return
    
    
    def get_gains(self):
        """
        Returns a list containing the last calculated gain of each leaf.
        """
        return [leaf.get_gain() for leaf in self.tree.get_leaves()]
        
        

class WorldModelTree(object):
    """
    This class implements the binary tree that is used in WorldModel to organize
    the state space partitions. It serves as super class for classes like
    WorldModelPCA that implement different methods for splitting the states.
    """

    def __init__(self, model, parents=None):
        
        # family relations of node
        self.model = model
        self.status = 'leaf'
        self.children = []
        self.dat_ref = []   # indices of data belonging to this node
        self.split_cache = None
        self.parents = []
        if parents is not None:
            self.parents = parents 


    def _init_test(self, action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves.
        """
        raise NotImplementedError("Use subclass like WorldModelSpectral instead.")


    def _test(self, x):
        """
        Tests to which child the data point x belongs
        """
        raise NotImplementedError("Use subclass like WorldModelSpectral instead.")


    def classify(self, x):
        """
        Returns the state that x belongs to according to the current model. If
        x is a matrix, a list is returned containing a integer state for every
        row.
        """

        # is x a matrix?
        if x.ndim > 1:

            # classify every point
            N = x.shape[0]
            labels = np.zeros(N)
            for i in range(N):
                labels[i] = self.classify(x[i])
            return labels

        else:
            
            status = self.status
            assert status in ['leaf', 'split', 'merged']

            if status == 'leaf':
                return self.get_leaf_index()
            elif status == 'split':
                child_index = int(self._test(x))
                return self.children[child_index].classify(x)
            elif status == 'merged':
                return self.children[0].classify(x)
            else:
                raise RuntimeError('Should not happen!')
            
            
    def _relabel_data(self):
        """
        Returns new labels and split data references according to the _test()
        method of a leaf node. So, _test() has to be initialized before but the
        node not finally split yet.
        """
        assert self.status == 'leaf'

        # some useful variables
        current_state = self.get_leaf_index()
        assert current_state is not None
        
        # make of copy of all labels
        # increase labels above current state by one to make space for the split
        new_labels = map(lambda l: l+1 if l > current_state else l, self.model.labels)
        new_dat_ref = [[], []]

        # every entry belonging to this node has to be re-classified
        for ref_i in self.dat_ref:
            dat = self.model.data[ref_i]
            child_i = self._test(dat)
            new_labels[ref_i] += child_i
            new_dat_ref[child_i].append(ref_i)

        assert len(new_labels) == len(self.model.labels)
        # does the split really split the data in two?
        #assert len(new_dat_ref[0]) > 0
        #assert len(new_dat_ref[1]) > 0
        if (len(new_dat_ref[0]) == 0 or
            len(new_dat_ref[1]) == 0):
            return None, None
        return new_labels, new_dat_ref


    def get_gain(self):
        if self.split_cache is None or not self.split_cache.has_key(None):
            self._calculate_best_split(action=None)
        if self.split_cache is not None and self.split_cache.has_key(None):
            return self.split_cache[None].gain
        else:
            return 0.0 


    def get_data(self):
        """
        Returns the data belonging to the node. If the node isn't a leaf, the
        data of sub-nodes is returned.
        """
        
        dat_refs = self._get_data_refs()
        if len(dat_refs) == 0:
            return None
        
        # fetch the actual data
        data_list = map(lambda i: self.model.data[i], dat_refs)
        return np.vstack(data_list)


    def _get_data_refs(self):
        """
        Returns the data references (i.e. indices for root.data) belonging to 
        the node. If the node isn't a leaf, the data of sub-nodes is returned.
        """

        if self.status == 'leaf':
        
            return self.dat_ref
        
        elif (self.status == 'split' or
              self.status == 'merged'):
            
            data_refs_set = set([])
            for child in self.children:
                data_refs_set = data_refs_set.union(child._get_data_refs())
            
            data_refs = list(data_refs_set)
            data_refs.sort()
            return data_refs
            
        else:            
            raise RuntimeError('Should not happen!')
        
        
    def _get_data_refs_for_action(self, action):
        """
        Returns the data references (i.e. indices for root.data) belonging to 
        the node. If the node isn't a leaf, the data of sub-nodes is returned.
        """

        if self.status == 'leaf':
        
            if self.model.actions is None:
                return self._get_data_refs()
            
            refs = self.dat_ref
            actions = self.model.actions
            
            return [t for t in refs if actions[t+1] == action]
        
        elif (self.status == 'split' or
              self.status == 'merged'):
            
            data_refs_set = set([])
            for child in self.children:
                data_refs_set = data_refs_set.union(child._get_data_refs())
            
            data_refs = list(data_refs_set)
            data_refs.sort()
            return data_refs
            
        else:            
            raise RuntimeError('Should not happen!')
        
        
    def _get_data_for_refs(self, refs):
        """
        Returns a data matrix for a list of references.
        """
        if len(refs) == 0:
            return None
        data_list = [self.model.data[t] for t in refs]
        data = np.vstack(data_list)
        return data
    
    
    def get_leaf_index(self):
        """
        Returns an integer class label for a leaf-node. If the node isn't a 
        leaf, 'None' is returned.
        """
        return self.get_root().get_leaves().index(self)
    
    
    def get_root(self):
        """
        Returns the root node of the whole tree.
        """
        if len(self.parents) == 0:
            return self
        else:
            return self.parents[0].get_root()


    def get_leaves(self):
        """
        Returns a list of all leaves belonging to the node.
        """
        if self.status == 'leaf':
            return [self]
        elif (self.status == 'split' or
              self.status == 'merged'):
            children = []
            for child in self.children:
                for new_child in child.get_leaves():
                    if new_child not in children:
                        children += [new_child]
            return children
        
        
    def _get_nodes(self):
        """
        Returns a list of all nodes.
        """
        nodes = set([self])
        for child in self.children:
            nodes.add(child)
            nodes = nodes.union(child._get_nodes())
        return nodes


    def _calculate_best_split(self, action=None):
        """
        Calculates the gain in mutual information if this node would be split.
        
        Calculations can be restricted to a given action.
        """
        
        # return cached result
        if self.split_cache is not None and self.split_cache.has_key(action):
            return self.split_cache[action]
        self.split_cache = {}

        # prepare action list
        if self.model.actions is None:
            action_list = [None]
        else:
            if action is None:
                action_list = set(self.model.transitions.keys())
                action_list.remove(None)
            else:
                action_list = [action]
        
        best_gain = float('-Inf')
        
        for a in action_list:
            
            print 'testing leaf', self.get_leaf_index(), 'with action', a
            best_gain_for_action = float('-Inf')
            
            for fast_partition in [False, True]:
                
                try:
                    
                    if self._init_test(action=a, fast_partition=fast_partition):
                        
                        gain = self._calc_local_gain(action=a)
                        if gain is None:
                            print 'USELESS SPLIT'
                            continue
                        
                        split = SplitResult(node = self,
                                            action = a, 
                                            gain = gain, 
                                            classifier = self.classifier)
                            
                        if gain > best_gain:
                            best_gain = gain
                            self.split_cache[None] = split                         
                        if gain > best_gain_for_action:
                            best_gain_for_action = gain
                            self.split_cache[a] = split
                        
                    else:
                        print 'init_test failed'
                        
                except Exception as e:
                    print 'Error calculating splitting gain'
                    print e
                    print traceback.print_exc()
                    
        return self.split_cache[None] if self.split_cache.has_key(None) else None
    
    
    def _apply_split(self, split_result):
        """
        Splits the node.
        """
        
        print 'splitting...'
        assert self.status == 'leaf'
        
        # copy split result
        self.split_gain = split_result.gain
        self.classifier = split_result.classifier
        new_labels, new_data_refs = self._relabel_data()
        self.model.transitions = self.model._split_transition_matrices(new_labels=new_labels, 
                                                                       index1=self.get_leaf_index())
        self.model.labels = new_labels
        
        # create new leaves
        child0 = self.__class__(model=self.model, parents = [self])
        child1 = self.__class__(model=self.model, parents = [self])
        child0.dat_ref = new_data_refs[0]
        child1.dat_ref = new_data_refs[1]
        
        # create list of children
        self.children = []
        self.children.append(child0)
        self.children.append(child1)
        self.status = 'split'   # make it official!

        # initialize a first split
        child0._calculate_best_split()
        child1._calculate_best_split()
        self.split_cache = None
        return
    
    
    def _get_transition_refs(self, heading_in=False, inside=True, heading_out=True):
        """
        Finds all transitions that start, end or happen strictly inside the 
        node. The result is given as two lists of references. One for the start 
        and one for the end of the transition.
        """
        refs = self._get_data_refs()
        N = self.model.get_number_of_samples()
        
        refs_in = [] 
        refs_inside = [] 
        refs_out = []
         
        if heading_in:
            refs_in = [ref-1 for ref in refs if (ref-1 not in refs) and (ref > 0)]
            
        if inside:
            refs_inside = [ref for ref in refs if (ref+1 in refs)]

        if heading_out:
            refs_out = [ref for ref in refs if (ref+1 not in refs) and (ref+1 < N)]
            
        refs_1 = list(np.sort(refs_in + refs_inside + refs_out))
        refs_2 = [t+1 for t in refs_1]
        
        return [refs_1, refs_2]
        
        
    def _get_transition_refs_for_action(self, action, heading_in=False, inside=True, heading_out=True):
        """
        Finds all transitions that start in the node. The result is given as two
        lists of references. One for the start and one for the end of the
        transition.
        """
        if self.model.actions is None:
            return self._get_transition_refs()
        
        _, r2 = self._get_transition_refs(heading_in=heading_in, inside=inside, heading_out=heading_out)
        refs_2 = [r for r in r2 if self.model.actions[r] == action]
        refs_1 = [r-1 for r in refs_2]
        return [refs_1, refs_2]
        
        
    def _reached_min_sample_size(self, action=None):
        
        if action is None:
            action_list = self.model.transitions.keys()
        else:
            action_list = [action]
            
        for action in action_list:
            if action is None:
                continue
            refs, _ = self._get_transition_refs_for_action(action=action)
            print 'number of samples for leaf', self.get_leaf_index(), 'action', action, ':', len(refs)
            if len(refs) < self.model._min_class_size:
                return False
            
        return True


    def _calc_local_gain(self, action):
        """
        Calculates the mutual entropy for a bipartition of that node (call 
        _init_test() before).
        """
        return self.model._mutual_information(self._calc_local_transition_matrix(action))
    
    
    def _calc_local_transition_matrix(self, action):
        """
        Calculates a (2x2-) transition matrix for the bi-partition induced by
        this node (call _init_test() before!).  
        """
        N = len(self.dat_ref)
        trans = np.zeros((2,2))
        child_indices = [self._test(self.model.data[ref]) for ref in self.dat_ref]
        for i in range(N-1):
            if self.model.actions is None or self.model.actions[self.dat_ref[i+1]] == action:
                c1 = child_indices[i]
                c2 = child_indices[i+1]
                trans[c1,c2] += 1
        if np.sum(trans[0]) == 0:
            return None
        if np.sum(trans[1]) == 0:
            return None
        return trans
        
    
# class WorldModelTreeOld(object):
# 
#     symbols = ['o', '^', 'd', 's', '*']
# 
#     def __init__(self, parents=None):
#         
#         # family relations of node
#         self.status = 'leaf'
#         self.children = []
#         self.parents = []
#         if parents is not None:
#             self.parents = parents 
#         
#         # attributes of root node
#         self.data = None     # global data storage in root node
#         self.transitions = None
#         self.actions = None     # either None or a list of actions
#         self.random = random.Random()
#         #self.random.seed(1)
#         self._min_class_size = 100
#         self.last_gain = 0
#         
#         # data of leaf
#         self.dat_ref = []    # indices of data belonging to this node
#         self.stats = []
#         
# 
#     def classify(self, x):
#         """
#         Returns the state that x belongs to according to the current model. If
#         x is a matrix, a list is returned containing a integer state for every
#         row.
#         """
# 
#         # is x a matrix?
#         if x.ndim > 1:
# 
#             # classify every point
#             N = x.shape[0]
#             labels = np.zeros(N)
#             for i in range(N):
#                 labels[i] = self.classify(x[i])
#             return labels
# 
#         else:
#             
#             status = self.status
#             assert status in ['leaf', 'split', 'merged']
# 
#             if status == 'leaf':
#                 return self.get_leaf_index()
#             elif status == 'split':
#                 child_index = int(self._test(x))
#                 return self.children[child_index].classify(x)
#             elif status == 'merged':
#                 return self.children[0].classify(x)
#             else:
#                 raise RuntimeError('Should not happen!')
#             
#             
#     def classify_to_vector(self, x):
#         """
#         Returns an indicator vector [.. 0 1 0 ..] instead of an integer class
#         label. Can be useful for instance, to perform a regression analysis on
#         the states. If x is a matrix, a matrix will be returned with one
#         indicator vector in each row.
#         """
#         
#         K = self.get_number_of_states()
#         
#         # is x a matrix?
#         if x.ndim > 1:
# 
#             labels = self.classify(x)
#             N = x.shape[0]
#             Y = np.zeros((N, K))
#             for i in range(N):
#                 s = labels[i]
#                 Y[i,s] = 1
# 
#         else:
#             
#             s = self.classify(x)
#             Y = np.ones(K)
#             Y[s] = 1
#             
#         return Y
#                 
# 
#     def _relabel_data(self):
#         """
#         Returns new labels and split data references according to the _test()
#         method of a leaf node. So, _test() has to be initialized before but the
#         node not finally split yet.
#         """
#         assert self.status == 'leaf'
# 
#         # some useful variables
#         root = self.root()
#         current_state = self.get_leaf_index()
#         assert current_state is not None
#         
#         # make of copy of all labels
#         # increase labels above current state by one to make space for the split
#         new_labels = map(lambda l: l+1 if l > current_state else l, root.labels)
#         new_dat_ref = [[], []]
# 
#         # every entry belonging to this node has to be re-classified
#         for ref_i in self.dat_ref:
#             dat = root.data[ref_i]
#             child_i = self._test(dat)
#             new_labels[ref_i] += child_i
#             new_dat_ref[child_i].append(ref_i)
# 
#         assert len(new_labels) == len(root.labels)
#         # does the split really split the data in two?
#         #assert len(new_dat_ref[0]) > 0
#         #assert len(new_dat_ref[1]) > 0
#         if (len(new_dat_ref[0]) == 0 or
#             len(new_dat_ref[1]) == 0):
#             return None, None
#         return new_labels, new_dat_ref
# 
# 
#     @classmethod
#     def _split_transition_matrices(cls, root, new_labels, index1, index2=None):
#         """
#         Calculates a split transition matrix for each action. The result is
#         a dictionary with actions as keys.
#         """
#         
#         N = len(new_labels)
#         result = {}
# 
#         # check
#         transitions = root._merge_transition_matrices()
#         #print np.sum(transitions), '==', N-1
#         assert np.sum(transitions) == N-1
#         
#         for action in root.transitions.keys():
#             result[action] = cls._split_transition_matrix(root=root, action=action, new_labels=new_labels, index1=index1, index2=index2)
#             
#         return result
# 
# 
#     @classmethod
#     def _split_transition_matrix(cls, root, action, new_labels, index1, index2=None):
#         """
#         Calculates a new transition matrix with the split index1 -> index1 & index1+1.
#         
#         In special cases it might be necessary to have a split index1 -> index1 & index2.
#         """
#         N = len(new_labels)
#         
#         assert root.get_leaves()[index1].status == 'leaf'
#         #print  max(new_labels),   root.transitions[action].shape[0]
#         assert max(new_labels) == root.transitions[action].shape[0]
#         if index2 is None:
#             index2 = index1
#             assert root.get_leaves()[index1].status == 'leaf'
# 
#         # new transition matrix
#         new_trans = np.array(root.transitions[action])
#         # split current row and set to zero
#         new_trans[index1,:] = 0
#         new_trans = np.insert(new_trans, index2, 0, axis=0)  # new row
#         # split current column and set to zero
#         new_trans[:,index1] = 0
#         new_trans = np.insert(new_trans, index2, 0, axis=1)  # new column
#         
#         # update all transitions from or to current state
#         for i in range(N-1):
#             source = root.labels[i]
#             target = root.labels[i+1]
#             if root.actions is None or root.actions[i+1] == action:
#                 if source == index1 or target == index1:
#                     new_source = new_labels[i]
#                     new_target = new_labels[i+1]
#                     new_trans[new_source, new_target] += 1
#         
#         return new_trans
#     
#     
#     def get_number_of_states(self):
#         """Returns the number of states in the model."""
#         return len(self.root().get_leaves())
#     
#     
#     def get_number_of_samples(self):
#         """Returns the number of data points stored in the model."""
#         if self is self.root():
#             return self.data.shape[0]
#         else:
#             return len(self.get_refs())
#         
#         
#     def get_input_dim(self):
#         """
#         Returns the input dimensionality of the model.
#         """
#         return self.root().data.shape[1]
#     
#     
#     def get_possible_actions(self, ignore_none=False):
#         """
#         Returns a set of actions used so far. In principle other actions can
#         be added afterwards with new data.
#         """
#         root = self.root()
#         actions = set(root.transitions.keys())
#         if ignore_none and root.actions is not None and None in actions:
#             actions.remove(None)
#         return actions
# 
# 
#     def add_data(self, x, actions=None):
#         """
#         Adds a matrix x of new observations to the node. The data is
#         interpreted as one observation following the previous one. This is
#         important to calculate the transition probabilities.
#         
#         The actions are a interpreted as actions that *preceded* each step. If
#         you don't know the action that preceded the first data point, it's okay
#         to leave the list of actions shorter by one. The missing action will be 
#         filled with 'None' and the transaction ignored during most calculations.
#         
#         If there has been data before, the new data is appended.
#         """
# 
#         # add data to root node only
#         root = self.root()
#         if self is not root:
#             root.add_data(x)
#             return
#         
#         # check for dimensionality of x
#         x = np.atleast_2d(x)
# 
#         # initialize actions
#         n = x.shape[0]
#         #if actions is None:
#         #    actions = [None for _ in range(n)]
#         if actions and len(actions) < n:
#             actions = [None] + actions
#         assert actions is None or len(actions) == n
# 
#         # calculate labels for new data
#         labels = np.empty(n, dtype=int)
#         for i in range(n):
#             labels[i] = self.classify(x[i])
#             
#         # store data to root node
#         if self.data is None:
#             first_data = 0
#             first_source = 0
#             self.data = x
#             self.labels = labels
#             self.actions = actions
#         else:
#             first_data = self.data.shape[0]
#             first_source = first_data - 1
#             self.data = np.vstack([self.data, x])
#             self.labels = np.hstack([self.labels, labels])
#             if self.actions is not None:
#                 self.actions = self.actions + actions
#             
#         # same number of actions and data points
#         assert actions is None or self.data.shape[0] == len(self.actions)
#         assert self.data.shape[0] == self.labels.shape[0]
# 
#         # add references to data in all leaves
#         all_leaves = self.get_leaves()
#         N = self.data.shape[0]
#         for i in range(first_data, N):
#             state = self.labels[i]
#             leaf  = all_leaves[state]
#             leaf.dat_ref.append(i)
#                 
#         # create global transition matrices (for each action)
#         K = len(all_leaves)
#         if self.transitions is None:
#             self.transitions = {}
#         action_set = [None] if self.actions is None else set(actions).union([None])
#         for action in action_set:
#             if action not in self.transitions.keys():
#                 self.transitions[action] = np.zeros((K,K))
#             
#         # update transition matrices
#         for i in range(first_source, N-1):
#             source = self.labels[i]
#             target = self.labels[i+1]
#             action = None
#             if self.actions is not None:
#                 action = self.actions[i+1]
#             self.transitions[action][source, target] += 1
# 
#         return
#     
# 
#     def get_leaves(self):
#         """
#         Returns a list of all leaves belonging to the node.
#         """
#         if self.status == 'leaf':
#             return [self]
#         elif (self.status == 'split' or
#               self.status == 'merged'):
#             children = []
#             for child in self.children:
#                 for new_child in child.get_leaves():
#                     if new_child not in children:
#                         children += [new_child]
#             return children
#         
#         
#     def _nodes(self):
#         """
#         Returns a list of all nodes.
#         """
#         nodes = set([self])
#         for child in self.children:
#             nodes.add(child)
#             nodes = nodes.union(child._nodes())
#         return nodes
# 
# 
#     def root(self):
#         """
#         Returns the root node of the whole tree.
#         """
#         if len(self.parents) == 0:
#             return self
#         else:
#             return self.parents[0].root()
# 
# 
#     def get_leaf_index(self):
#         """
#         Returns an integer class label for a leaf-node. If the node isn't a 
#         leaf, 'None' is returned.
#         """
#         return self.root().get_leaves().index(self)
#     
#     
#     def get_most_interesting_leaf(self):
#         """
#         Returns the node that contributed the highest gain when it was split.
#         """
#         root = self.root()
#         leaves = root.get_leaves()
#         gains = [leave.last_gain for leave in leaves]
#         max_i = np.argmax(gains)
#         return leaves[max_i]
# 
# 
#     def plot_states(self, show_plot=True, range_x=None, range_y=None, resolution=100):
#         """
#         Shows a contour plot of the learned states (2D). 
#         """
#         
#         root = self.root()
#         data = root.get_data()
#         if data.shape[1] > 2:
#             data = sklearn.manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(data)
#         K = len(root.get_leaves())
#         
#         if range_x is None:
#             range_x = [np.min(data[:,0]), np.max(data[:,0])]
#             
#         if range_y is None:
#             range_y = [np.min(data[:,1]), np.max(data[:,1])]
#             
#         x = np.linspace(range_x[0], range_x[1], resolution)
#         y = np.linspace(range_y[0], range_y[1], resolution)
#         X, Y = np.meshgrid(x, y)
#         v_classify = np.vectorize(lambda x, y: self.classify(np.array([x,y])))
#         Z = v_classify(X, Y)
#         pyplot.contourf(X, Y, Z, levels = range(-1, K))
#         
#         if show_plot:
#             pyplot.show()
#         return
#          
# 
#     def plot_state_borders(self, show_plot=True, range_x=None, range_y=None, resolution=100):
#         """
#         Shows a contour plot of the learned state borders (2D). 
#         """
#         
#         root = self.root()
#         data = root.get_data()
#         if data.shape[1] > 2:
#             data = sklearn.manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(data)
#         K = len(root.get_leaves())
#         
#         if range_x is None:
#             range_x = [np.min(data[:,0]), np.max(data[:,0])]
#             
#         if range_y is None:
#             range_y = [np.min(data[:,1]), np.max(data[:,1])]
#             
#         x = np.linspace(range_x[0], range_x[1], resolution)
#         y = np.linspace(range_y[0], range_y[1], resolution)
#         X, Y = np.meshgrid(x, y)
#         v_classify = np.vectorize(lambda x, y: self.classify(np.array([x,y])))
#         Z = v_classify(X, Y)
#         pyplot.contour(X, Y, Z, levels = range(-1, K), colors='b', linewidths=1)
#         
#         if show_plot:
#             pyplot.show()
#         return
#          
# 
#     def plot_tree_data(self, color='state', vmin=None, vmax=None, ndim=None, show_plot=True):
#         """
#         Plots all the data that is stored in the tree with color and shape
#         according to the learned state.
#         """
#         
#         if color == 'state':
# 
#             # list of data for the different classes
#             data_list = []
#             all_leaves = self.get_leaves()
#             for leaf in all_leaves:
#                 data = leaf.get_data()
#                 if data is not None:
#                     if ndim is not None and ndim < self.get_input_dim():
#                         data = sklearn.manifold.Isomap(n_neighbors=10, n_components=ndim).fit_transform(data)
#                     data_list.append(data)
#     
#             # plot
#             colormap = pyplot.cm.get_cmap('prism')
#             pyplot.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.98, 7)])
#             for i, data in enumerate(data_list):
#                 pyplot.plot(data[:,0], data[:,1], self.symbols[i%len(self.symbols)])
#                 
#         elif color == 'last_gain':
#             
#             leaves = self.get_leaves()
#             if vmin is None:
#                 vmin = min([leaf.last_gain for leaf in leaves])
#             if vmax is None:
#                 vmax = max([leaf.last_gain for leaf in leaves])
#             colormap = pyplot.cm.get_cmap('summer')
#             
#             for leaf in leaves:
#                 data = leaf.get_data()
#                 if ndim is not None and ndim < self.get_input_dim():
#                     data = sklearn.manifold.Isomap(n_neighbors=10, n_components=ndim).fit_transform(data)
#                 colors = [leaf.last_gain for _ in range(data.shape[0])]
#                 pyplot.scatter(x=data[:,0], y=data[:,1], c=colors, cmap=colormap, edgecolors='none', vmin=vmin, vmax=vmax)
#                 
#         else:
#             
#             data = self.root().get_data()
#             if ndim is not None and ndim < self.get_input_dim():
#                 data = sklearn.manifold.Isomap(n_neighbors=10, n_components=ndim).fit_transform(data)
#             pyplot.plot(data[:,0], data[:,1], '.')
#             
#         if show_plot:
#             pyplot.show()
#         return
#     
#     
#     def plot_stats(self, show_plot=True):
#         """
#         Plots how the models benchmark values have developed during training.
#         """
#         root = self.root()
#         stats = np.vstack(root.stats)
#         pyplot.plot(stats)
#         pyplot.legend(list(root.stats[0]._fields)[0:], loc=2)
#         
#         if show_plot:
#             pyplot.show()
#         return
# 
# 
#     def _get_data_refs(self):
#         """
#         Returns the data references (i.e. indices for root.data) belonging to 
#         the node. If the node isn't a leaf, the data of sub-nodes is returned.
#         """
# 
#         if self.status == 'leaf':
#         
#             return self.dat_ref
#         
#         elif (self.status == 'split' or
#               self.status == 'merged'):
#             
#             data_refs_set = set([])
#             for child in self.children:
#                 data_refs_set = data_refs_set.union(child._get_data_refs())
#             
#             data_refs = list(data_refs_set)
#             data_refs.sort()
#             return data_refs
#             
#         else:            
#             raise RuntimeError('Should not happen!')
#         
#         
#     def _get_data_refs_for_action(self, action):
#         """
#         Returns the data references (i.e. indices for root.data) belonging to 
#         the node. If the node isn't a leaf, the data of sub-nodes is returned.
#         """
# 
#         if self.status == 'leaf':
#         
#             root = self.root()
#             if root.actions is None:
#                 return self._get_data_refs()
#             
#             refs = self.dat_ref
#             actions = root.actions
#             
#             return [t for t in refs if actions[t+1] == action]
#         
#         elif (self.status == 'split' or
#               self.status == 'merged'):
#             
#             data_refs_set = set([])
#             for child in self.children:
#                 data_refs_set = data_refs_set.union(child._get_data_refs())
#             
#             data_refs = list(data_refs_set)
#             data_refs.sort()
#             return data_refs
#             
#         else:            
#             raise RuntimeError('Should not happen!')
#         
#         
#     def _get_transition_refs(self, heading_in=False, inside=True, heading_out=True):
#         """
#         Finds all transitions that start, end or happen strictly inside the 
#         node. The result is given as two lists of references. One for the start 
#         and one for the end of the transition.
#         """
#         refs = self._get_data_refs()
#         N = self.root().get_number_of_samples()
#         
#         refs_in = [] 
#         refs_inside = [] 
#         refs_out = []
#          
#         if heading_in:
#             refs_in = [ref-1 for ref in refs if (ref-1 not in refs) and (ref > 0)]
#             
#         if inside:
#             refs_inside = [ref for ref in refs if (ref+1 in refs)]
# 
#         if heading_out:
#             refs_out = [ref for ref in refs if (ref+1 not in refs) and (ref+1 < N)]
#             
#         refs_1 = list(np.sort(refs_in + refs_inside + refs_out))
#         refs_2 = [t+1 for t in refs_1]
#         
#         return [refs_1, refs_2]
#         
#         
#     def _get_transition_refs_for_action(self, action, heading_in=False, inside=True, heading_out=True):
#         """
#         Finds all transitions that start in the node. The result is given as two
#         lists of references. One for the start and one for the end of the
#         transition.
#         """
#         root = self.root()
#         if root.actions is None:
#             return self._get_transition_refs()
#         
#         _, r2 = self._get_transition_refs(heading_in=heading_in, inside=inside, heading_out=heading_out)
#         refs_2 = [r for r in r2 if root.actions[r] == action]
#         refs_1 = [r-1 for r in refs_2]
#         return [refs_1, refs_2]
#         
#         
#     def _get_data_for_refs(self, refs):
#         """
#         Returns a data matrix for a list of references.
#         """
#         root = self.root()
#         if len(refs) == 0:
#             return None
#         data_list = [root.data[t] for t in refs]
#         data = np.vstack(data_list)
#         return data
#     
#     
#     def _get_actions_for_refs(self, refs):
#         """
#         Returns a list of actions that preceeded each of the given data points
#         (given as a list of references).
#         """
#         root = self.root()
#         if len(refs) == 0:
#             return None
#         action_list = [root.actions[t] for t in refs]
#         return action_list
#         
#                 
#     def get_data(self):
#         """
#         Returns the data belonging to the node. If the node isn't a leaf, the
#         data of sub-nodes is returned.
#         """
#         
#         dat_refs = self._get_data_refs()
#         if len(dat_refs) == 0:
#             return None
#         
#         # fetch the actual data
#         root = self.root()
#         data_list = map(lambda i: root.data[i], dat_refs)
#         return np.vstack(data_list)
# 
# 
#     @classmethod
#     def _entropy(cls, dist, normalize=False, ignore_empty_classes=False):
#         """
#         Calculates the (normalized) entropy over a given probability 
#         distribution.
#         """
# 
#         # negative values?
#         assert True not in list(dist < -0.)
# 
#         # useful variables
#         trans_sum = np.sum(dist)
#         K = len(dist)
# 
#         # only one class?
#         if K <= 1:
#             return 1.0
# 
#         # empty class?
#         if trans_sum == 0:
#             if not ignore_empty_classes:
#                 assert trans_sum != 0
#             if normalize:
#                 return 1.0
#             else:
#                 return np.log2(K)
# 
#         # the actual calculation
#         probs = np.array(dist, dtype=np.float64) / trans_sum
#         log_probs = np.zeros_like(probs)
#         log_probs[probs > 0.] = np.log2( probs[probs > 0.] )
#         entropy = -np.sum(probs * log_probs)
# 
#         # normalization?
#         assert(entropy <= np.log2(K) + 1e-8)
#         if normalize:
#             entropy /= np.log2(K)
# 
#         assert(entropy >= 0)
#         return entropy
# 
# 
#     def entropy(self):
#         """
#         Calculates the entropy for all training data, i.e. the transition 
#         entropy of each state weighted by how often it occurs.
#         """        
#         transitions = self._merge_transition_matrices()
#         return self._matrix_entropy(transitions=transitions, normalize=False)
# 
# 
#     @classmethod
#     def _matrix_entropy(cls, transitions, normalize=False):
#         """
#         Calculates the entropy for a given transition matrix, i.e. the 
#         transition entropy of each state weighted by how often it occurs.
#         """        
#         K = transitions.shape[0]
#         row_entropies = np.zeros(K)
# 
#         for i in range(K):
#             row = transitions[i]
#             row_entropies[i] = cls._entropy(dist=row, normalize=normalize, ignore_empty_classes=True)
# 
#         # weighted average
#         weights = np.sum(transitions, axis=1)
#         weights /= np.sum(weights)
#         entropy = np.sum(weights * row_entropies)
#         return entropy
#     
#     
#     def _merge_transition_matrices(self, transitions=None):
#         """
#         Merges the transition matrices of each action to a single one.
#         """
#         
#         root = self.root()
#         if transitions is None:
#             transitions = root.transitions
#         if transitions is None:
#             return None
#         K = transitions.itervalues().next().shape[0]
#         
#         P = np.zeros((K,K))
#         for action in root.transitions.keys():
#             P += root.transitions[action]
#             
#         return P
#     
#     
# #    def split(self, action):
# #        """
# #        Splits all leaves belonging to that node.
# #        """
# #        
# #        print 'splitting...'
# #        # recursion to leaves
# #        if self.status != 'leaf':
# #            for leaf in self.leaves():
# #                leaf.split(action)
# #            return
# #        
# #        # test for minimum number of data points
# #        assert self.status == 'leaf'
# #        #if len(self.dat_ref) < self._min_class_size:
# #        #if len(self.get_data_strict()) < self._min_class_size:
# #        #    return
# #        
# #        root = self.root()
# #        if len(self.parents) == 2: # a leaf that was just merged
# #            
# #            # a split would be redundant here because the current node was just 
# #            # merged. so 'simply' revert that merging...
# #
# #            parent1 = self.parents[0]
# #            parent2 = self.parents[1]
# #            self.parents = []
# #            parent1.status = 'leaf'
# #            parent2.status = 'leaf'
# #            parent1.children = []
# #            parent2.children = []
# #            label1 = parent1.class_label()
# #            label2 = parent2.class_label()
# #
# #            # make of copy of all labels
# #            # increase labels above second node by one to make space for the split
# #            new_labels = map(lambda l: l+1 if l >= label2 else l, root.labels)
# #            new_dat_ref = [[], []]
# #    
# #            # every entry belonging to this node has to be re-classified
# #            for ref_i in self.dat_ref:
# #                dat = root.data[ref_i]
# #                label = root.classify(dat)
# #                assert label == label1 or label == label2
# #                if label == label1:
# #                    new_dat_ref[0].append(ref_i)
# #                else:
# #                    new_dat_ref[1].append(ref_i)
# #                    new_labels[ref_i] = label2
# #                    
# #            assert len(new_labels) == len(root.labels)
# #            root.transitions = self._split_transition_matrices(root=root, new_labels=new_labels, index1=label1, index2=label2)
# #            root.labels = new_labels
# #            
# #            parent1.dat_ref = new_dat_ref[0]
# #            parent2.dat_ref = new_dat_ref[1]
# #            assert len(parent1.dat_ref) + len(parent2.dat_ref) == len(self.dat_ref)
# #            
# #            print 'TRIVIAL SPLIT'
# #            
# #        else:
# #            
# #            # re-classify data
# #            self._init_test(action=action)
# #            new_labels, new_dat_ref = self._relabel_data()
# #
# #            #print [np.sum(m) for m in root.transitions.itervalues()]
# #            root.transitions = self._split_transition_matrices(root=root, new_labels=new_labels, index1=self.get_class_label())
# #            #print [np.sum(m) for m in root.transitions.itervalues()]
# #            root.labels = new_labels
# #            
# #            # create new leaves
# #            child0 = self.__class__(parents = [self])
# #            child1 = self.__class__(parents = [self])
# #            child0.dat_ref = new_dat_ref[0]
# #            child1.dat_ref = new_dat_ref[1]
# #    
# #            # create list of children
# #            self.children = []
# #            self.children.append(child0)
# #            self.children.append(child1)
# #            self.status = 'split'
# #            
# #        return
#     
#     
#     def _apply_split(self, split_result):
#         """
#         Splits all leaves belonging to that node.
#         """
#         
#         print 'splitting...'
#         assert self.status == 'leaf'
#         root = self.root()
# 
#         # copy split result
#         self.classifier = split_result.classifier
#         root.transitions = split_result.split_transitions
#         root.labels = split_result.split_labels
#         
#         # create new leaves
#         child0 = self.__class__(parents = [self])
#         child1 = self.__class__(parents = [self])
#         child0.dat_ref = split_result.split_data_refs[0]
#         child1.dat_ref = split_result.split_data_refs[1]
#         
#         # initialize last_gain with values of parent
#         child0.last_gain = split_result.gain
#         child1.last_gain = split_result.gain
#         
#         # create list of children
#         self.children = []
#         self.children.append(child0)
#         self.children.append(child1)
#         self.status = 'split'
#             
#         return
#     
#     
#     def _calculate_best_split(self, action=None):
#         """
#         Calculates the gain in mutual information if this node would be split.
#         
#         Calculations can be restricted to a given action.
#         
#         TODO: cache result!
#         """
#         
#         root = self.root()
#         assert self in root.get_leaves()
#         best_gain = float('-Inf')
#         best_split = None
#         
#         if root.actions is None:
#             action_list = [None]
#         else:
#             if action is None:
#                 action_list = set(root.transitions.keys())
#                 action_list.remove(None)
#             else:
#                 action_list = [action]
#         
#         for action in action_list:
#             if root.transitions[action].sum() < root._min_class_size:
#                 continue
#             print 'testing leaf', self.get_leaf_index(), 'with action', action
#             for fast_partition in [False, True]:
#                 try:
#                     if self._init_test(action=action, fast_partition=fast_partition):
#                         new_labels, new_data = self._relabel_data()
#                         if new_labels is None:
#                             print 'USELESS SPLIT'
#                             #assert False
#                             continue
#                         split_transition_matrices = self._split_transition_matrices(root=root, new_labels=new_labels, index1=self.get_leaf_index())
#                         new_mutual_information = self._mutual_information_average(transition_matrices=split_transition_matrices.values())
#                         old_mutual_information = self._mutual_information_average(transition_matrices=root.transitions.values()) # TODO cache
#                         gain = new_mutual_information - old_mutual_information
#                         if gain > best_gain:
#                             best_gain = gain
#                             best_split = SplitResult(node = self, 
#                                                      action = action, 
#                                                      gain = gain, 
#                                                      split_labels = new_labels, 
#                                                      split_data_refs = new_data,
#                                                      split_transitions = split_transition_matrices, 
#                                                      classifier = self.classifier)
#                     else:
#                         print 'init_test failed'
#                 except Exception as e:
#                     print 'Error calculating splitting gain'
#                     print e
#                     print traceback.print_exc()
#                     #tkMessageBox.showinfo(title='Exception', message='%s' % e)
# 
#         return best_split
#     
#     
#     def single_splitting_step(self, action=None, min_gain=float('-inf')):
#         """
#         Calculates the gain for each state and splits the best one. Can be
#         restricted to a given action.
#         
#         TODO: only re-calculate states with some change
#         """
#         
#         root = self.root()
#         assert self is root
#         best_gain = float('-inf')
#         best_split = None
#         
#         for leaf in self.root().get_leaves():
#             print 'testing leaf', leaf.get_leaf_index(), '...'
#             split = leaf._calculate_best_split(action=action)
#             if split is not None and leaf._reached_min_sample_size(action=action):
#                 print 'best split: leaf', leaf.get_leaf_index(), 'with action', split.action, 'with gain', split.gain
#                 leaf.last_gain = split.gain
#                 if split.gain > best_gain:
#                     best_gain = split.gain
#                     best_split = split
#                 
#         if best_split is not None and best_gain >= min_gain:
#             print 'decided for leaf', best_split.node.get_leaf_index(), 'with action', best_split.action, 'and gain', best_split.gain
#             best_split.node._apply_split(split_result=best_split)
#             root.stats.append(self._calc_stats(transitions=root.transitions))
#             
#         return best_gain
#     
#     
#     def learn(self, action=None, min_gain=0.02):
#         """
#         Learns the model.
#         """
#         root = self.root()
#         assert self is root
#         
#         # init stats
#         if len(root.stats) == 0:
#             root.stats.append(self._calc_stats(transitions=root.transitions))
# 
#         # split as long as it's interesting            
#         #self.single_splitting_step(min_gain=float('-inf'))
#         gain = float('inf')
#         while gain >= min_gain:
#             gain = self.single_splitting_step(action=action, min_gain=min_gain)
#             if gain >= min_gain:
#                 print 'split with gain', gain, '\n'
#             
#         return gain
#     
#     
#     def _merge_nodes(self, s1, s2):
#         """
#         Merges two nodes.
#         """
#         
#         root = self.root()
#         leaves = root.leaves()
#         leaf1 = leaves[s1]
#         leaf2 = leaves[s2]
#         assert leaf1.status == 'leaf'
#         assert leaf2.status == 'leaf'
#         
#         # merge transitions
#         root.transitions = self._merge_matrix(root.transitions, s1, s2)
#         
#         # merge labels
#         for i in range(len(root.labels)):
#             if root.labels[i] == s2:
#                 root.labels[i] = s1
#             if root.labels[i] > s2:
#                 root.labels[i] -= 1
# 
#         if leaf1.parents[0] == leaf2.parents[0]: # same parent
#             
#             # trivial merge: revert split of parent
#             # merge data references and set new status
#             parent = leaf1.parents[0]
#             parent.dat_ref = leaf1.dat_ref + leaf2.dat_ref
#             parent.children = []
#             parent.status = 'leaf'
#             print 'TRIVIAL MERGE'
# 
#         else:
#         
#             # merge data references
#             leaves = root.leaves()
#             parent1 = leaves[s1]            
#             parent2 = leaves[s2]            
#             child = self.__class__(parents = [parent1, parent2])
#             child.dat_ref = parent1.dat_ref + parent2.dat_ref
#             parent1.dat_ref = []
#             parent2.dat_ref = []
#             parent1.children = [child]
#             parent2.children = [child]
#             parent1.status = 'merged'
#             parent2.status = 'merged'
#             
#         return
#     
#     
#     @classmethod
#     def _merge_matrix(cls, matrix, s1, s2):
#         """
#         Merges rows and columns of a matrix.
#         """
#         matrix[s1,:] += matrix[s2,:]
#         matrix = np.delete(matrix, s2, 0)  
#         matrix[:,s1] += matrix[:,s2]
#         matrix = np.delete(matrix, s2, 1)
#         return matrix
#     
#     
#     def _init_test(self, action, fast_partition=False):
#         """
#         Initializes the parameters that split the node in two halves.
#         """
#         raise NotImplementedError("Use subclass like WorldModelSpectral instead.")
# 
# 
#     def _test(self, x):
#         """
#         Tests to which child the data point x belongs
#         """
#         raise NotImplementedError("Use subclass like WorldModelSpectral instead.")
# 
# 
#     @classmethod
#     def _mutual_information(cls, transition_matrix):
#         """
#         Calculates the mutual information between t and t+1 for a model given
#         as transition matrix. 
#         """
#         P = transition_matrix
#         assert np.sum(P) > 0
#         weights = np.sum(P, axis=1)
#         mu = weights / np.sum(weights)
#         entropy_mu = cls._entropy(dist=mu)
#         entropy = cls._matrix_entropy(transitions=transition_matrix)
#         mutual_information = entropy_mu - entropy
#         return mutual_information
#     
#     
#     @classmethod
#     def _mutual_information_average(cls, transition_matrices):
#         """
#         Calculates the weighted average of mutual information for several models 
#         given as a list of transition matrices.
#         """
#         weights = [np.sum(P) for P in transition_matrices]
#         list_mi = [weights[i] * cls._mutual_information(P) for i, P in enumerate(transition_matrices) if weights[i] > 0]
#         return np.sum(list_mi) / np.sum(weights)
#             
#     
#     def _reached_min_sample_size(self, action=None):
#         
#         if action is None:
#             action_list = self.root().transitions.keys()
#         else:
#             action_list = [action]
#             
#         for action in action_list:
#             if action is None:
#                 continue
#             refs, _ = self._get_transition_refs_for_action(action=action)
#             print 'number of samples for leaf', self.get_leaf_index(), 'action', action, ':', len(refs)
#             if len(refs) < self._min_class_size:
#                 return False
#             
#         return True
#         
#     
#     def _calc_stats(self, transitions):
#         """
#         Calculates statistics for a given transition matrix.
#         """
# 
#         n_nodes = len(self.root()._nodes())
#         
#         P = self._merge_transition_matrices(transitions)
#         K = P.shape[0]
#         entropy = self._matrix_entropy(transitions=P)
#         entropy_normalized = self._matrix_entropy(transitions=P, normalize=True)
#         
#         # norm of Q
#         weights = np.sum(P, axis=1)
#         probs = P / weights[:,np.newaxis]
#         mu = weights / np.sum(weights)
#         norm = np.sum( ( probs**2 * mu[:,np.newaxis] ) / mu[np.newaxis,:] )
#         
#         # mutual information
#         entropy_mu = self._entropy(dist=mu)
#         mutual_information = entropy_mu - entropy
# 
#         stats = Stats(n_states = K,
#                       n_nodes = n_nodes, 
#                       entropy = entropy, 
#                       entropy_normalized = entropy_normalized, 
#                       norm = norm, 
#                       mutual_information = mutual_information)
#         return stats
#     
#     
#     def get_transition_probabilities(self, action=None, soft=False):
#         """
#         Returns a dictionary containing for every action a matrix with 
#         transition probabilities. If an action is specified only the 
#         corresponding transition matrix is returned.
#         """
#         root = self.root()
#         
#         if action not in root.transitions.keys():
#             return None
#         
#         if root.actions is not None and action is None:
#             probs = {}
#             for action in root.transitions.keys():
#                 probs[action] = np.array(root.transitions[action])
#                 if soft:
#                     probs[action] += 1
#                 probs[action] /= probs[action].sum(axis=1)[:, np.newaxis] # normalize
#                 
#         else:
#             probs = np.array(root.transitions[action]) 
#             if soft:
#                 probs[action] += 1
#             probs /= probs.sum(axis=1)[:, np.newaxis]
#         
#         return probs
#     
#     
#     def get_graph_cost_matrix(self, soft=False):
#         """
#         Returns a distance matrix for the learned states that may be useful for
#         calculating a shortest path through the state space.
#         
#         The distances are calculated like this: first, we are considering the 
#         probability of moving from one state to another after selecting the best 
#         action for that transition. For that probability we then calculate the 
#         (negative) logarithm. The Dijkstra algorithm for instance works by 
#         summing up distances which would results in the (log-) product of our 
#         probability values.
#         """
#         
#         # helper variables
#         root = self.root()
#         if root.get_number_of_samples() <= 1:
#             return np.ones((1,1))
#         probs = self.get_transition_probabilities(soft=soft)
#         actions = probs.keys()
#         if root.actions is not None and None in actions:
#             actions.remove(None)
#         num_actions = len(actions)
#         num_states = self.get_number_of_states()
#         
#         # tensor that holds all transition matrices
#         T = np.zeros((num_actions, num_states, num_states))
#         for i, action in enumerate(actions):
#             T[i] = probs[action]
#             
#         # from transition probabilities to affinity
#         A = T.max(axis=0)
#         A = -np.log(A)
#         return A
#     
#     
#     def get_state_means(self):
#         """
#         Returns a matrix with mean values for every state.
#         """
#         
#         root = self.root()
#         K = self.get_number_of_states()
#         D = self.get_input_dim()
#         means = np.zeros((K, D))
#         
#         for i, state in enumerate(root.get_leaves()):
#             means[i] = state.get_data().mean(axis=0)
#             
#         return means
# 
#     
#     
# # class WorldModelMeta(object):
# #     
# #     
# #     def __init__(self):
# #         self.models = {}
# #         self.last_action_added = None
# #     
# #     
# #     def add_data(self, x, actions=None):
# #         
# #         # initialize list of actions
# #         N = x.shape[0]
# #         if actions is not None:
# #             if len(actions) < N:
# #                 actions = [None] + actions
# #             assert len(actions) == N
# #             
# #         # add first row
# #         if actions is not None:
# #             action = actions[0]
# #             if action not in self.models.keys():
# #                 self.models[action] = WorldModelTree()
# #             self.models[action].add_data(x=x[0], actions=[action])
# #         else:
# #             action = None
# #             if action not in self.models.keys():
# #                 self.models[action] = WorldModelTree()
# #             self.models[action].add_data(x=x[0], actions=None)
# #         self.last_action_added = action
# #         
# #         # add each row to corresponding model
# #         for i in range(1, N):
# #             
# #             if actions is None:
# # 
# #                 self.models[None].add_data(x=x[i], actions=None)
# #                 
# #             else:
# #                 
# #                 action = actions[i]
# #                 if action != self.last_action_added:
# #                     if action not in self.models.keys():
# #                         self.models[action] = WorldModelTree()
# #                     self.models[action].add_data(x[i-1], actions=[None])
# #                     
# #                 self.models[action].add_data(x[i], actions=[action])
# #                 self.last_action_added = action
# #                         
# #         return
# #     
# #     
# #     def learn(self, min_gain=0.02):
# #         for action in self.models:
# #             self.models[action].learn(min_gain=min_gain)
# #         return



class WorldModelPCA(WorldModelTree):    

    
    def _init_test(self, action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves.
        """
        if fast_partition:
            return False
        
        assert self.status == 'leaf'
        
        refs_1, refs_2 = self._get_transition_refs_for_action(action=action, heading_in=False, inside=True, heading_out=False)
        refs = list(set(refs_1 + refs_2))
        data = self._get_data_for_refs(refs=refs)
        avg = np.mean(data, axis=0)
        data_0 = data - avg
        cov = data_0.T.dot(data_0)
        E, U = np.linalg.eigh(cov)
        idx = np.argsort(E)
        u = U[:,idx[-1]]
        self.classifier = (avg, u)
        return True


    def _test(self, x):
        """
        Tests to which child the data point x belongs
        """
        if x.ndim < 2:
            x = np.array(x, ndmin=2)
        y = x - self.classifier[0]
        z = y.dot(self.classifier[1])
        return 0 if z <= 0 else 1
        
        
    
class WorldModelSpectral(WorldModelTree):
    
    
    def __init__(self, **kwargs):
        super(WorldModelSpectral, self).__init__(**kwargs)


    def _get_transition_graph(self, action=None, k=15, fast_partition=False, normalize=True):
        assert self.status == 'leaf'
        assert action in self.model.get_possible_actions(ignore_none=False)
        
        # data and references
        [refs_1, refs_2] = self._get_transition_refs_for_action(action=action, heading_in=False, inside=True, heading_out=False)
        data = self._get_data_for_refs(refs_1)
        refs_all = list(set(refs_1 + refs_2))
        n_trans_all = len(refs_all)
        n_trans = len(refs_1)        
        if n_trans <= 0:
            return [], [], None
        
        # pairwise distances
        distances = scipy.spatial.distance.pdist(data)
        distances = scipy.spatial.distance.squareform(distances)
        
        # transitions
        W = np.zeros((n_trans_all, n_trans_all))
        W += 0.00001

        # big transition matrix
        # adding transitions to the k nearest neighbors
        for i in range(n_trans):
            indices = np.argsort(distances[i])  # closest one should be the point itself
            # index: refs -> refs_all
            s = refs_all.index(refs_1[i])
            for j in indices[0:k+1]:
                # index: refs -> refs_all
                t = refs_all.index(refs_1[j])
                if s != t:
                    W[s,t] = 1
                    W[t,s] = 1
        
        # big transition matrix
        # adding transitions of the k nearest neighbors
        for i in range(n_trans):
            indices = np.argsort(distances[i])  # closest one should be the point itself
            # index: refs -> refs_all
            s = refs_all.index(refs_1[i])
            for j in indices[0:k+1]:
                # index: refs -> refs_all
                t = refs_all.index(refs_1[j])
                u = refs_all.index(refs_2[j])
                if s != t:
                    W[s,t] = 1
                    W[t,s] = 1
                if fast_partition:
                    W[s,u] = -1
                    W[u,s] = -1
                else:
                    W[s,u] = 1
                    W[u,s] = 1

        # make symmetric        
        #W = W + W.T
        #P = W + W.T
        P = W
        
        # normalize matrix
        if normalize:
            d = np.sum(P, axis=1)
            for i in range(n_trans_all):
                if d[i] > 0:
                    P[i] = P[i] / d[i]
            
        return refs_all, refs_1, P


    def _init_test(self, action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves.
        """
        assert self.status == 'leaf'

        # data        
        refs_all, refs_1, P = self._get_transition_graph(action=action, k=15, fast_partition=fast_partition, normalize=True)
        data = self._get_data_for_refs(refs=refs_1)
        n_trans = len(refs_1)
        
        # second eigenvector
        E, U = scipy.sparse.linalg.eigs(np.array(P), k=2, which='LR')
        E, U = np.real(E), np.real(U)
        
        # bi-partition
        if fast_partition:
            #idx = np.argsort(abs(E))
            idx = np.argsort(E)
            col = idx[-1]
        else:
            #idx = np.argsort(abs(E))
            idx = np.argsort(E)
            col = idx[-2]
        u = np.zeros(n_trans)
        for i in range(n_trans):
            # index: refs -> refs_all
            row = refs_all.index(refs_1[i])
            u[i] = U[row,col].real
        u -= np.mean(u)
        #assert -1 in np.sign(u)
        #assert 1 in np.sign(u)
        if -1 not in np.sign(u):
            return False
        if 1 not in np.sign(u):
            return False
        
        # classifier
        labels = map(lambda x: 1 if x > 0 else 0, u)
        self.classifier = mdp.nodes.KNNClassifier(k=20)
        #self.classifier = mdp.nodes.NearestMeanClassifier()
        #self.classifier = mdp.nodes.LibSVMClassifier(probability=False)
        self.classifier.train(data, np.array(labels, dtype='int'))
        self.classifier.stop_training()
        y = self.classifier.label(data)
        if 0 not in y:
            return False
        if 1 not in y:
            return False
        return True


    def _test(self, x):
        """
        Tests to which child the data point x belongs
        """
        if x.ndim < 2:
            x = np.array(x, ndmin=2)
        return int(self.classifier.label(x)[0])
    
    
    
class WorldModelSFA(WorldModelTree):
    
    def _init_test(self, action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves.
        """
        assert self.status == 'leaf'
        
        # data
        refs_1, refs_2 = self._get_transition_refs_for_action(action=action, heading_in=False, inside=True, heading_out=False)
        refs = np.sort(list(set(refs_1 + refs_2)))
        data = self._get_data_for_refs(refs=refs)
        _, D = data.shape
        
        # SFA
        self.classifier = mdp.Flow([])
        
        for _ in range(1):
            mdp_exp = mdp.nodes.PolynomialExpansionNode(degree=2)
            mdp_pca = mdp.nodes.PCANode(svd=True)
            mdp_sfa = mdp.nodes.SFANode(output_dim=4)
            self.classifier += mdp.Flow([mdp_exp, mdp_pca, mdp_sfa])
        
        mdp_exp = mdp.nodes.PolynomialExpansionNode(degree=2)
        mdp_pca = mdp.nodes.PCANode(svd=True)
        mdp_sfa = mdp.nodes.SFANode(output_dim=D)
        self.classifier += mdp.Flow([mdp_exp, mdp_sfa])
        
        self.classifier.train(data)
        
        # 
        if fast_partition:
            self.classifier[-1].sf = self.classifier[-1].sf[:,::-1]
            self.classifier[-1].d = self.classifier[-1].d[::-1]
            self.classifier[-1]._bias = self.classifier[-1]._bias[::-1]
        
        # verify solution
        labels = np.sign(self.classifier.execute(data)[:,0])
        if 1 not in labels:
            return False
        if -1 not in labels:
            return False
        return True


    def _test(self, x):
        """
        Tests to which child the data point x belongs
        """
        if x.ndim < 2:
            x = np.array(x, ndmin=2)
        signal = self.classifier.execute(x)[0,0]
        return 0 if signal < .5 else 1
    
    

class SimpleIterable(object):
    def __init__(self, blocks):
        self.blocks = blocks
    def __iter__(self):
        # this is a generator
        for block in self.blocks:
            yield block

    
class WorldModelGraphSFA(WorldModelTree):
    
    def _get_transition_graph(self, action=None, k=10, normalize=True):
        assert self.status == 'leaf'
        assert action in self.get_possible_actions(ignore_none=False)
        
        # data and references
        [refs_1, refs_2] = self._get_transition_refs_for_action(action=action, heading_in=False, inside=True, heading_out=False)
        data = self._get_data_for_refs(refs_1)
        refs_all = list(set(refs_1 + refs_2))
        n_trans_all = len(refs_all)
        n_trans = len(refs_1)        
        if n_trans <= 0:
            return [], [], None
        
        # pairwise distances
        distances = scipy.spatial.distance.pdist(data)
        distances = scipy.spatial.distance.squareform(distances)
        
        # transitions
        W = np.zeros((n_trans_all, n_trans_all))
        #W += 0.001
        
        # big transition matrix
        # including transitions of the k nearest neighbors
        for i in range(n_trans):
            indices = np.argsort(distances[i])  # closest one should be the point itself
            # index: refs -> refs_all
            s = refs_all.index(refs_1[i])
            for j in indices[0:k+1]:
                # index: refs -> refs_all
                #t = refs_all.index(refs_1[j])
                u = refs_all.index(refs_2[j])
                #W[s,t] = 0.1
                #W[t,s] = 0.1
                W[s,u] = 1
                W[u,s] = 1

        # make symmetric        
        #W = W + W.T
        P = (W + W.T) / 2.
        
        # normalize matrix
        if normalize:
            d = np.sum(P, axis=1)
            for i in range(n_trans_all):
                if d[i] > 0:
                    P[i] = P[i] / d[i]
            
        return refs_all, refs_1, P
    
    
    def _init_test(self, action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves.
        """
        if fast_partition:
            return False
        
        assert self.status == 'leaf'
        
        # get data and graph
        refs_all, _, W = self._get_transition_graph(action=action, k=10, normalize=False)
        print refs_all
        data = self._get_data_for_refs(refs=refs_all)
        N, D = data.shape
        
        # create training chunks according to connection graph
        chunks = []
        for i in range(N):
            for j in range(i+1, N):
                if W[i,j] >= 1:
                    chunk = np.array([data[i], data[j]])
                    chunks.append(chunk)
        train_iterable = SimpleIterable(chunks)
                    
        # (repeated) graph-based SFA
        self.classifier = mdp.Flow([])
        train_list = []
        for _ in range(2):
            mdp_exp = mdp.nodes.PolynomialExpansionNode(degree=2)
            mdp_sfa = mdp.nodes.SFANode(output_dim=2*D, include_last_sample=False)
            self.classifier += mdp.Flow([mdp_exp, mdp_sfa])
            train_list.append(None)
            train_list.append(train_iterable)
            
        # train SFA
        self.classifier.train(train_list)
        
        # verify solution
        labels = np.sign(self.classifier.execute(data)[:,0])
        if 1 not in labels:
            return False
        if -1 not in labels:
            return False
        return True


    def _test(self, x):
        """
        Tests to which child the data point x belongs
        """
        if x.ndim < 2:
            x = np.array(x, ndmin=2)
        signal = self.classifier.execute(x)[0,0]
        return 0 if signal < .5 else 1

    

class WorldModelFactorize():
    """
    This meta-model organizes one model for each action to allow for different
    (and hopefully orthogonal) partitions for each action.
    """
   
    def __init__(self):
        self.models = {}
    
    
    def add_data(self, data, actions):
        new_action_set = set(actions)
        known_actions = set(self.models.keys())
        for action in known_actions.union(new_action_set):
            # add model if action was not known before
            if action not in self.models.keys():
                self.models[action] = WorldModel(method='factorize')
            self.models[action].add_data(data, actions=actions)
        return
    
    
    def learn(self, min_gain=0.02):
        for action in self.models.keys():
            self.models[action].learn(action=action, min_gain=min_gain)
            
            
    def single_splitting_step(self, action=None, min_gain=float('-inf')):
        
        gain = 0
        
        if action is None:
            actions = self.models.keys()
        else:
            actions = [action]
            
        for a in actions:
            gain += self.models[a].single_splitting_step(action=a, min_gain=min_gain)
            
        return gain / len(actions)

    
    
class WorldModelFactorizeNode(WorldModelTree):
    """
    A model class that is suited to be used by WorldModelFactorize.
    """
    
    def _get_transition_graph(self, fast_action, k=15, normalize=True):

        refs = self._get_data_refs()
        data = self._get_data_for_refs(refs)
        actions = self._get_actions_for_refs(refs)
        N = len(refs)        
        
        # pairwise distances
        distances = scipy.spatial.distance.pdist(data)
        distances = scipy.spatial.distance.squareform(distances)
    
        # transition matrix
        W = np.zeros((N, N))
        #W += 1e-8
        
        # number of actions
        action_set = set(actions)
        if None in action_set:
            action_set.remove(None)
        n_actions = len(action_set)
        weight = n_actions - 1
    
        # transitions to neighbors
        # s - current node
        # t - neighbor node
        # u - following node
        for s in range(N):
            indices = np.argsort(distances[s])  # closest one should be the point itself
            for t in indices[0:k+1]:
                if s != t:
                    if actions[s] == fast_action:
                        W[s,t] = 1#weight
                        W[t,s] = 1#weight
                    else:
                        W[s,t] = 1
                        W[t,s] = 1
    
        # transitions to successors
        # s - current node
        # t - neighbor node
        # u - following node (of neighbor)
        for s, _ in enumerate(refs):
            indices = np.argsort(distances[s])  # closest one should be the point itself
            for t in indices[0:k+1]:
                ref_t = refs[t]
                ref_u = ref_t + 1
                if ref_u not in refs:
                    continue
                u = refs.index(ref_u)
                if actions[u] == fast_action:
                    W[s,u] = -weight
                    W[u,s] = -weight
                else:
                    W[s,u] = 1
                    W[u,s] = 1
                    
        # normalize matrix
        if normalize:
            d = np.sum(W, axis=1)
            for i in range(N):
                if d[i] > 0:
                    W[i] = W[i] / d[i]
                
        return W


    def _init_test(self, action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves.
        """
        assert self.status == 'leaf'
        
        if fast_partition:
            W = self._get_transition_graph(fast_action=action, k=15, normalize=True)
        else:
            return False
            
        # laplacian
        W = np.diag(np.sum(W, axis=1)) - W

        # data        
        refs = self._get_data_refs()
        data = self._get_data_for_refs(refs=refs)
        
        # second eigenvector
        E, U = scipy.linalg.eig(a=(W+W.T))
        #E, U = linalg.eigs(np.array(W), k=2, which='LR')
        E, U = np.real(E), np.real(U)
        
        # bi-partition
        idx = np.argsort(E)
        col = idx[0]
        u = U[:,col]
        #u -= np.mean(u)
        #assert -1 in np.sign(u)
        #assert 1 in np.sign(u)
        if -1 not in np.sign(u):
            return False
        if 1 not in np.sign(u):
            return False
        
        # classifier
        labels = map(lambda x: 1 if x > 0 else 0, u)
        self.classifier = mdp.nodes.KNNClassifier(k=20)
        #self.classifier = mdp.nodes.NearestMeanClassifier()
        #self.classifier = mdp.nodes.LibSVMClassifier(probability=False)
        self.classifier.train(data, np.array(labels, dtype='int'))
        self.classifier.stop_training()
        y = self.classifier.label(data)
        if 0 not in y:
            return False
        if 1 not in y:
            return False
        return True


    def _test(self, x):
        """
        Tests to which child the data point x belongs
        """
        if x.ndim < 2:
            x = np.array(x, ndmin=2)
        return int(self.classifier.label(x)[0])



def problemChain(n=1000, seed=None):
    """
    4 states in a line.
    """

    if seed is not None:
        np.random.seed(seed)

    data = []
    for _ in range(n//4+1):
        data.append(np.random.randn(2) + [-6,-3])
        data.append(np.random.randn(2) + [-2,-1])
        data.append(np.random.randn(2) + [+2, 1])
        data.append(np.random.randn(2) + [+6, 3])
    data = np.vstack(data)
    data = data[0:n]
    return data


def problemDiamond(n=1000, seed=None):
    """
    Transition between seven states in a honeycomb-like structure.
    """
    if seed is not None:
        np.random.seed(seed)

    data = []
    for _ in range(n//4+1):
        data.append(np.random.randn(2) + [-4, 0])
        data.append(np.random.randn(2) + [-0, 3])
        data.append(np.random.randn(2) + [+4, 0])
        data.append(np.random.randn(2) + [+0,-3])
    data = np.vstack(data)
    data = data[0:n]
    return data


def problemHoneycomb(n=1000, seed=None):
    """
    4 states in a circle.
    """
    dist = 6
    c = np.sin(math.pi/3)
    m1 = (-dist/2, dist*c)
    m2 = (+dist/2, dist*c)
    m3 = (-dist, 0)
    m4 = (0, 0)
    m5 = (dist, 0)
    m6 = (-dist/2, -dist*c)
    m7 = (+dist/2, -dist*c)
    means = [m1, m2, m4, m3, m5, m6, m7]

    if seed is not None:
        np.random.seed(seed)

    data = []
    for _ in range(n//7+1):
        for i in range(len(means)):
            data.append(np.random.randn(2) + [means[i][0], means[i][1]])

    data = np.vstack(data)
    data = data[0:n]
    return data


if __name__ == "__main__":
    
    print scipy.version.version

    problems = [problemChain, problemDiamond, problemHoneycomb]
    #problems = [problemHoneycomb]

    for p, problem in enumerate(problems):

        # create data
        n = 1000
        data = problem(n=n, seed=None)

        model = WorldModel()
        model.add_data(data)

        #print tree.transitions
        #tree.single_splitting_step()
        #tree.single_splitting_step()
        #tree.single_splitting_step()
        #tree.single_splitting_step()
        model.learn(min_gain=0.01)

        print model.transitions
        n_trans = np.sum(model._merge_transition_matrices())
        print 'final number of nodes:', len(model.tree._get_nodes()), '\n'
        print n_trans, '==', n-1
        assert(n_trans == n-1)

        # plot tree and stats
        pyplot.subplot(3, 3, p+1)
        model.plot_data(color='state', show_plot=False)
        pyplot.subplot(3, 3, p+4)
        model.plot_states(show_plot=False)
        pyplot.subplot(3, 3, p+7)
        model.plot_data(color='last_gain', show_plot=False)

    #pyplot.figure()
    #refs_1, refs_2 = tree.get_leaves()[1]._get_transition_refs_for_action(action=None)
    #data = tree.get_leaves()[0]._get_data_for_refs(refs=refs_1)
    #pyplot.plot(data[:,0], data[:,1], 'o')
    #data = tree.get_leaves()[0]._get_data_for_refs(refs=refs_2)
    #pyplot.plot(data[:,0], data[:,1], 'x')    
    pyplot.show()
    
