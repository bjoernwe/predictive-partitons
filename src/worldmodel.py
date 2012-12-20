import collections
import math
import numpy as np
import random

from matplotlib import pyplot

import mdp


Stats = collections.namedtuple('Stats', ['n_states',
                                         'n_nodes', 
                                         'norm', 
                                         'entropy', 
                                         'entropy_normalized', 
                                         'mutual_information'])

class WorldModelTree(object):

    symbols = ['o', '^', 'd', 's', '*']

    def __init__(self, parents=None):
        
        # family relations of node
        self.status = 'leaf'
        self.children = []
        self.parents = []
        if parents is not None:
            self.parents = parents 
        
        # attributes of root node
        self.data = None     # global data storage in root node
        self.transitions = None
        self.random = random.Random()
        #self.random.seed(1)
        self._min_class_size = 50
        
        # data of leaf
        self.dat_ref = []    # indices of data belonging to this node
        self.stats = []



    def classify(self, x):
        """
        Takes a single vector and returns an integer class label.
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

            if self.status == 'leaf':
                return self.class_label()
            elif self.status == 'split':
                child_index = int(self._test(x))
                class_label = self.children[child_index].classify(x)
                return class_label
            elif self.status == 'merged':
                class_label = self.children[0].classify(x)
                return class_label
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
        root = self.root()
        current_state = self.class_label()
        assert current_state is not None
        
        # make of copy of all labels
        # increase labels above current state by one to make space for the split
        new_labels = map(lambda l: l+1 if l > current_state else l, root.labels)
        new_dat_ref = [[], []]

        # every entry belonging to this node has to be re-classified
        for ref_i in self.dat_ref:
            dat = root.data[ref_i]
            child_i = self._test(dat)
            new_labels[ref_i] += child_i
            new_dat_ref[child_i].append(ref_i)

        assert len(new_labels) == len(root.labels)
        return new_labels, new_dat_ref


    @classmethod
    def _splitted_transition_matrix(cls, root, new_labels, index1, index2=None):
        """
        Calculates a new transition matrix with the split index1 -> index1 & index1+1.
        
        In special cases it might be necessary to have a split index1 -> index1 & index2.
        """

        N = len(new_labels)
        assert root.leaves()[index1].status == 'leaf'
        if index2 is None:
            index2 = index1
            assert root.leaves()[index1].status == 'leaf'

        # new transition matrix
        new_trans = np.array(root.transitions)
        # split current row and set to zero
        new_trans[index1,:] = 0
        new_trans = np.insert(new_trans, index2, 0, axis=0)  # new row
        # split current column and set to zero
        new_trans[:,index1] = 0
        new_trans = np.insert(new_trans, index2, 0, axis=1)  # new column

        # update all transitions from or to current state
        for i in range(N-1):
            source = root.labels[i]
            target = root.labels[i+1]
            if source == index1 or target == index1:
                new_source = new_labels[i]
                new_target = new_labels[i+1]
                new_trans[new_source, new_target] += 1

        assert np.sum(new_trans) == len(new_labels)-1
        return new_trans


    def add_data(self, x):
        """
        Adds a matrix x of new observations to the node. The data is
        interpreted as one observation following the previous one. This is
        important to calculate the transition probabilities.
        If there has been data before, the new data is appended.
        """

        # add data to root node only
        root = self.root()
        if self is not root:
            root.add_data(x)
            return

        # calculate labels for new data
        n = x.shape[0]
        labels = np.empty(n, dtype=int)
        for i in range(n):
            labels[i] = self.classify(x[i])

        # store data to root node
        if self.data is None:
            first_data = 0
            first_source = 0
            self.data = x
            self.labels = labels
        else:
            first_data = self.data.shape[0]
            first_source = first_data - 1
            self.data = np.vstack([self.data, x])
            self.labels = np.vstack([self.labels, labels])

        # add references to data in all leaves
        all_leaves = self.leaves()
        N = self.data.shape[0]
        for i in range(first_data, N):
            state = self.labels[i]
            leaf  = all_leaves[state]
            leaf.dat_ref.append(i)
                
        # create a global transition matrix            
        K = len(all_leaves)
        if self.transitions is None:
            self.transitions = np.zeros((K,K))
            
        # update transition matrix
        for i in range(first_source, N-1):
            source = self.labels[i]
            target = self.labels[i+1]
            self.transitions[source, target] += 1

        return
    

    def leaves(self):
        """
        Returns a list of all leaves belonging to the node.
        """
        if self.status == 'leaf':
            return [self]
        elif (self.status == 'split' or
              self.status == 'merged'):
            children = []
            for child in self.children:
                for new_child in child.leaves():
                    if new_child not in children:
                        children += [new_child]
            return children
        
        
    def nodes(self):
        """
        Returns a list of all nodes.
        """
        nodes = set([self])
        for child in self.children:
            nodes.add(child)
            nodes = nodes.union(child.nodes())
        return nodes


    def root(self):
        """
        Returns the root node of the whole tree.
        """
        if len(self.parents) == 0:
            return self
        else:
            return self.parents[0].root()


    def class_label(self):
        """
        Returns an integer class label for a leaf-node. If the node isn't a leaf,
        None is returned.
        """
        return self.root().leaves().index(self)


    def plot_tree_data(self, data_list=None, show_plot=True):
        """
        Plots all the data that is stored in the tree.
        """


        if data_list is None:
            # list of data for the different classes
            data_list = []
            all_leaves = self.leaves()
            for leaf in all_leaves:
                data = leaf.get_data()
                if data is not None:
                    data_list.append(data)

        # plot
        colormap = pyplot.cm.prism
        pyplot.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.98, 7)])
        for i, data in enumerate(data_list):
            pyplot.plot(data[:,0], data[:,1], self.symbols[i%len(self.symbols)])
            
        if show_plot:
            pyplot.show()
        return
    
    
    def plot_stats(self, show_plot=True):
        stats = np.vstack(self.root().stats)
        #pyplot.gca().invert_xaxis()
        #for i in range(1, stats.shape[1]):
        #    pyplot.plot(stats[:,0], stats[:,i])
        pyplot.plot(stats)
        pyplot.legend(list(tree.stats[0]._fields)[0:], loc=2)
        return


    def get_data_refs(self):
        """
        Returns the data references belonging to the node. If the node isn't a leaf, the
        data of sub-nodes is returned.
        """

        if self.status == 'leaf':
        
            return self.dat_ref
        
        elif (self.status == 'split' or
              self.status == 'merged'):
            
            data_set = set([])
            for child in self.children:
                data_set = data_set.union(child.get_data())
            
            data_refs = list(data_set)
            data_refs.sort()
            return data_refs
            
        else:            
            raise RuntimeError('Should not happen!')
        

    def get_data(self):
        """
        Returns the data belonging to the node. If the node isn't a leaf, the
        data of sub-nodes is returned.
        """
        
        dat_refs = self.get_data_refs()
        if len(dat_refs) == 0:
            return None
        
        # fetch the actual data
        root = self.root()
        data_list = map(lambda i: root.data[i], dat_refs)
        return np.vstack(data_list)


    @classmethod
    def _entropy(cls, trans, normalize=False, ignore_empty_classes=False):
        """
        Calculates the (normalized) entropy over the target state distribution
        for a leaf or a given list of transitions.
        """

        # negative values?
        assert True not in list(trans < -0.)

        # useful variables
        trans_sum = np.sum(trans)
        K = len(trans)

        # only one class?
        if K <= 1:
            return 1.0

        # empty class?
        
        if trans_sum == 0:
            #if not ignore_empty_classes:
            #    assert trans_sum != 0
            if normalize:
                return 1.0
            else:
                return np.log2(K)

        # the actual calculation
        probs = np.array(trans, dtype=np.float64) / trans_sum
        log_probs = np.zeros_like(probs)
        log_probs[probs > 0.] = np.log2( probs[probs > 0.] )
        entropy = -np.sum(probs * log_probs)

        # normalization?
        assert(entropy <= np.log2(K))
        if normalize:
            entropy /= np.log2(K)

        assert(entropy >= 0)
        return entropy


    def entropy(self):
        """
        Calculates the (weighted) entropy for the root node.
        """
        return self._matrix_entropy(transitions=self.root().transitions, normalize=False)


    @classmethod
    def _matrix_entropy(cls, transitions, normalize=False):
        """
        Calculates the entropy for a given transition matrix.
        """        
        K = transitions.shape[0]
        row_entropies = np.zeros(K)

        for i in range(K):
            row = transitions[i]
            row_entropies[i] = cls._entropy(trans=row, normalize=normalize)

        # weighted average
        weights = np.sum(transitions, axis=1)
        weights /= np.sum(weights)
        entropy = np.sum(weights * row_entropies)
        return entropy
    
    
    def split(self):
        """
        Splits all leaves belonging to that node.
        """
        
        # recursion to leaves
        if self.status != 'leaf':
            for leaf in self.leaves():
                leaf.split()
            return
        
        # test for minimum number of data points
        assert self.status == 'leaf'
        if len(self.dat_ref) < self._min_class_size:
            return
        
        root = self.root()
        if len(self.parents) == 2: # merged but same grandparent?
            
            # a split would be redundant here because the current node was just merged.
            # so simply revert that merging...

            parent1 = self.parents[0]
            parent2 = self.parents[1]
            parent1.status = 'leaf'
            parent2.status = 'leaf'
            label1 = parent1.class_label()
            label2 = parent2.class_label()

            # make of copy of all labels
            # increase labels above second node by one to make space for the split
            new_labels = map(lambda l: l+1 if l > label2 else l, root.labels)
            new_dat_ref = [[], []]
    
            # every entry belonging to this node has to be re-classified
            for ref_i in self.dat_ref:
                dat = root.data[ref_i]
                label = root.classify(dat)
                assert label == label1 or label == label2
                if label == label1:
                    new_dat_ref[0].append(ref_i)
                else:
                    new_labels[ref_i] = label2
                    new_dat_ref[1].append(ref_i)
                    
            assert len(new_labels) == len(root.labels)
            root.transitions = self._splitted_transition_matrix(root=root, new_labels=new_labels, index1=label1, index2=label2)
            root.labels = new_labels
            
            parent1.dat_ref = new_dat_ref[0]
            parent2.dat_ref = new_dat_ref[1]
            assert len(parent1.dat_ref) + len(parent2.dat_ref) == len(self.dat_ref)
            
        else:
            
            # re-classify data
            self._init_test()
            new_labels, new_dat_ref = self._relabel_data()
            root.transitions = self._splitted_transition_matrix(root=root, new_labels=new_labels, index1=self.class_label())
            root.labels = new_labels
            
            # create new leaves
            child0 = self.__class__(parents = [self])
            child1 = self.__class__(parents = [self])
            child0.dat_ref = new_dat_ref[0]
            child1.dat_ref = new_dat_ref[1]
    
            # create list of children
            self.children = []
            self.children.append(child0)
            self.children.append(child1)
            self.status = 'split'
            
        return
    
    
    def calculate_splitting_gain(self):
        """
        Calculates the gain in mutual information if this node would be splitted.
        
        TODO: cache result!
        """
        root = self.root()
        assert self in root.leaves()
        self._init_test()
        new_labels, _ = self._relabel_data()
        splitted_transition_matrix = self._splitted_transition_matrix(root=root, new_labels=new_labels, index1=self.class_label())
        new_mutual_information = self._mutual_information(transition_matrix=splitted_transition_matrix)
        old_mutual_information = self._mutual_information(transition_matrix=root.transitions)
        return new_mutual_information - old_mutual_information
    
    
    def single_splitting_step(self, min_gain=0):
        """
        Calculates the gain for each state and splits the best one.
        
        TODO: only re-calculate states with some change
        """
        
        assert self is self.root()
        best_leaf = None
        best_gain = float('-inf')
        
        for leaf in self.root().leaves():
            gain = leaf.calculate_splitting_gain()
            if gain >= best_gain:
                best_gain = gain
                best_leaf = leaf
                
        if best_leaf is not None and best_gain >= min_gain:
            best_leaf.split()
            root = self.root()
            root.stats.append(self._calc_stats(transitions=root.transitions))
            
        return best_gain
    
    
    def learn(self, min_gain=0.02, max_costs=0.02):
        """
        Learns the model.
        """
        root = self.root()
        assert self is root
        
        self.single_splitting_step(min_gain=float('-inf'))
        best_gain = float('inf')
        while best_gain >= min_gain:
            best_gain = self.single_splitting_step(min_gain=min_gain)
            print 'split with gain', best_gain 
            
        root.stats.append(self._calc_stats(transitions=root.transitions))
            
        # merge again ...
        for r in range(150):
        
            # useful variables    
            K = root.transitions.shape[0]
            if K < 2:
                break
            
            # find best merge ...
            best_s1 = None
            best_s2 = None
            best_costs = float('inf')
            
            if r % 100 == 0:    
                print 'round:', r
                
            for _ in range(250):

                # pick random pair of states for merging            
                s1, s2 = root.random.sample(range(K), 2)
                if s1 > s2:
                    s1, s2 = [s2, s1]
                
                # merge rows and columns
                merged_trans = np.array(root.transitions)
                merged_trans[s1,:] += merged_trans[s2,:]
                merged_trans = np.delete(merged_trans, s2, 0)  
                merged_trans[:,s1] += merged_trans[:,s2]
                merged_trans = np.delete(merged_trans, s2, 1)
                
                new_stats = self._calc_stats(transitions=merged_trans)
                costs = root.stats[-1].mutual_information - new_stats.mutual_information
                
                if (costs < best_costs):
                    
                    print 'best costs:', costs

                    best_s1 = s1
                    best_s2 = s2
                    best_costs = costs
                    best_stats = new_stats
                    
                    merged_trans = None
                    new_stats = None

            if best_s1 is None:
                continue
            
            if best_costs <= max_costs:
                self._merge_nodes(best_s1, best_s2)
                root.stats.append(best_stats)
                print 'merged:', best_costs
                print root.transitions
                print np.sum(root.transitions)
            else:
                return
                
        return
    
    
    def _merge_nodes(self, s1, s2):
        """
        Merges two nodes.
        """
        
        root = self.root()
        leaves = root.leaves()
        leaf1 = leaves[s1]
        leaf2 = leaves[s2]
        assert leaf1.status == 'leaf'
        assert leaf2.status == 'leaf'
        
        # merge transitions
        root.transitions[s1,:] += root.transitions[s2,:]
        root.transitions = np.delete(root.transitions, s2, 0)  
        root.transitions[:,s1] += root.transitions[:,s2]
        root.transitions = np.delete(root.transitions, s2, 1)
        
        # merge labels
        for i in range(len(root.labels)):
            if root.labels[i] == s2:
                root.labels[i] = s1
            if root.labels[i] > s2:
                root.labels[i] -= 1

        if leaf1.parents[0] == leaf2.parents[0]: # same parent
            
            # trivial merge: revert split of parent
            # merge data references and set new status
            parent = leaf1.parents[0]
            parent.dat_ref = leaf1.dat_ref + leaf2.dat_ref
            parent.children = []
            parent.status = 'leaf'

        else:
        
            # merge data references
            leaves = root.leaves()
            parent1 = leaves[s1]            
            parent2 = leaves[s2]            
            child = self.__class__(parents = [parent1, parent2])
            child.dat_ref = parent1.dat_ref + parent2.dat_ref
            parent1.dat_ref = []
            parent2.dat_ref = []
            parent1.children = [child]
            parent2.children = [child]
            parent1.status = 'merged'
            parent2.status = 'merged'
            
        return

        
    def _init_test(self):
        """
        Initializes the parameters that split the node in two halves.
        """
        assert self.status == 'leaf'
        # calculate PCA of data
        data = self.get_data()
        self.pca = mdp.nodes.PCANode(output_dim=1)
        self.pca.train(data)
        self.pca.stop_training()
        return
    
    
    def _test(self, x):
        """
        Tests to which child the data point x belongs
        """
        if x.ndim < 2:
            x = np.array(x, ndmin=2)
        index = self.pca.execute(x)[0,0]
        index = np.sign(index) + 1
        index = index // 2
        return int(index)
    
    
    def _mutual_information(self, transition_matrix):
        P = transition_matrix
        weights = np.sum(P, axis=1)
        mu = weights / np.sum(weights)
        entropy_mu = self._entropy(trans=mu)
        entropy = self._matrix_entropy(transitions=transition_matrix)
        mutual_information = entropy_mu - entropy
        return mutual_information
        
    
    def _calc_stats(self, transitions):
        """
        Calculates statistics for a given transition matrix.
        """

        n_nodes = len(self.root().nodes())
        
        P = transitions
        K = P.shape[0]
        entropy = self._matrix_entropy(transitions=P)
        entropy_normalized = self._matrix_entropy(transitions=P, normalize=True)
        
        # norm of Q
        weights = np.sum(P, axis=1)
        probs = P / weights[:,np.newaxis]
        mu = weights / np.sum(weights)
        norm = np.sum( ( probs**2 * mu[:,np.newaxis] ) / mu[np.newaxis,:] )
        
        # mutual information
        entropy_mu = self._entropy(trans=mu)
        mutual_information = entropy_mu - entropy

        stats = Stats(n_states = K,
                      n_nodes = n_nodes, 
                      entropy = entropy, 
                      entropy_normalized = entropy_normalized, 
                      norm = norm, 
                      mutual_information = mutual_information)
        return stats



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

    problems = [problemChain, problemDiamond, problemHoneycomb]
    #problems = [problemHoneycomb]

    for p, problem in enumerate(problems):

        # create data
        n = 10000
        data = problem(n=n, seed=1)

        tree = WorldModelTree()
        tree.add_data(data)


        print tree.transitions
        #tree.learn(min_gain=0.03, max_costs=0.03)
        tree.learn(min_gain=0.03, max_costs=0.05)


        n_trans = np.sum(tree.transitions)
        entropy = tree.entropy()
        print 'final transitions:\n', tree.transitions
        print 'sum:', n_trans
        print 'final entropy:', entropy
        print 'final number of nodes:', len(tree.nodes())
        assert(n_trans == n-1)

        # plot tree and stats
        pyplot.subplot(2, 3, p+1)
        tree.plot_tree_data(show_plot=False)
        pyplot.subplot(2, 3, p+3+1)
        tree.plot_stats(show_plot=False)

    pyplot.show()
