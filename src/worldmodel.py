import math
import numpy as np
import random

from matplotlib import pyplot

import mdp


class WorldModelTree(object):

    symbols = ['o', '^', 'd', 's', '*']

    def __init__(self, normalized_entropy, global_entropy, parents=None):
        
        # family relations of node
        self.status = 'leaf'
        self.parents = parents 
        self.children = []
        
        # attributes of root node
        self.data = None     # global data storage in root node
        self.transitions = None
        self._normalized_entropy = normalized_entropy
        self._global_entropy = global_entropy
        
        # data of leaf
        self.dat_ref = []    # indices of data belonging to this node

        #self._split_entropy = split_entropy
        #self._min_class_size = 10
        #self._min_n_states = 1
        #self._tmp_entropy = float('inf')
        #self._tmp_rating = float('-inf')


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


#    def _prepare_all_splits(self):
#        """
#        """
#
#        # recursion: step down to the leafes
#        if self._is_splitted():
#            for child in self.children:
#                child._prepare_all_splits()
#            return
#
#        # check wether there's enough data
#        # TODO parameterize
#        if len(self.dat_ref) < 2 * self._min_class_size:
#            return 0
#
#        # find best split ...
#        root = self.root()
#        best_rating = float('Inf')
#        for _ in range(100):
#
#            # initialize decision boundary of _test() method
#            if self._init_test() == False:
#                continue
#
#            # calculate labels and data for initialized split
#            new_labels, new_dat_ref = self._relabel_data()
#            if (len(new_dat_ref[0]) < self._min_class_size or 
#                len(new_dat_ref[1]) < self._min_class_size):
#                continue
#            
#            # calculate transitions matrix for new labels
#            new_trans = self._splitted_transition_matrix(new_labels)
#
#            # remember best split
#            new_entropy = self._transition_entropy(trans=new_trans, normalized_entropy=root._normalized_entropy, global_entropy=root._global_entropy)
#
#            # correct by entropy of the split itself
#            new_rating = new_entropy
#            if self._split_entropy:
#                split_sizes = np.zeros(2)
#                split_sizes[0] = len(new_dat_ref[0])
#                split_sizes[1] = len(new_dat_ref[1])
#                new_rating = new_entropy / self._entropy(trans=split_sizes, normalized_entropy=root._normalized_entropy)
#
#            if new_rating < best_rating:
#                best_rating = new_rating
#                best_entropy = new_entropy
#                best_labels = new_labels
#                best_refs = new_dat_ref
#                best_trans = new_trans
#                best_test = self._get_test_parameters()
#            
#        if best_rating == float('inf'):
#            return
#        self._tmp_rating  = best_rating
#        self._tmp_entropy = best_entropy
#        self._tmp_labels  = best_labels
#        self._tmp_refs    = best_refs
#        self._tmp_trans   = best_trans
#        self._tmp_test    = best_test
#        return
        
        
#    def split(self, force=False):
#        
#        root = self.root()
#        root._prepare_all_splits()
#        entropy = root.entropy()
#        all_leaves = root.leaves()
#        gains = map(lambda leaf: entropy - leaf._tmp_entropy, all_leaves)
#        #gains = map(lambda leaf: entropy - leaf._tmp_rating, all_leaves)
#        best_gain_index = np.argmax(gains)
#        
#        if len(all_leaves) >= self._min_n_states:
#            if (gains[best_gain_index] <= 0 or
#                gains[best_gain_index] == float('inf')):
#                return False
#        
#        # split!
#        best_leaf = all_leaves[best_gain_index]
#        best_leaf._set_test_parameters(best_leaf._tmp_test)
#        root.labels = best_leaf._tmp_labels
#        root.transitions = best_leaf._tmp_trans
#
#        # create new leaves
#        child0 = self.__class__(normalized_entropy = self._normalized_entropy, 
#                                global_entropy = self._global_entropy,
#                                split_entropy = self._split_entropy,
#                                parent = best_leaf)
#        child1 = self.__class__(normalized_entropy = self._normalized_entropy, 
#                                global_entropy = self._global_entropy,
#                                split_entropy = self._split_entropy,
#                                parent = best_leaf)
#        child0.dat_ref = best_leaf._tmp_refs[0]
#        child1.dat_ref = best_leaf._tmp_refs[1]
#
#        # create list of children
#        # (makes split official!)
#        best_leaf.children = []
#        best_leaf.children.append(child0)
#        best_leaf.children.append(child1)
#        
#        # reset calculated splits
#        for leaf in all_leaves:
#            leaf._tmp_rating = float('-inf')
#        
#        return True
                

    def _relabel_data(self):
        """
        Returns new labels and split data refs according to the _test()
        method of a leaf node. So, _test() has to be initialized before but the
        node not finally split yet.
        """

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


    def _splitted_transition_matrix(self, new_labels):
        """
        Calculates a new transition matrix with the current state split.
        """
        if not self.status == 'leaf':
            return None

        # helpful variable
        root = self.root()
        current_state = self.class_label()
        assert current_state is not None
        N = len(new_labels)

        # new transition matrix
        new_trans = np.array(root.transitions)
        # split current row and set to zero
        new_trans[current_state,:] = 0
        new_trans = np.insert(new_trans, current_state, 0, axis=0)  # new row
        # split current column and set to zero
        new_trans[:,current_state] = 0
        new_trans = np.insert(new_trans, current_state, 0, axis=1)  # new column

        # update all transitions from or to current state
        for i in range(N-1):
            source = root.labels[i]
            target = root.labels[i+1]
            if source == current_state or target == current_state:
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


    def root(self):
        """
        Returns the root node of the whole tree.
        """
        if self.parents is None:
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


    def _get_data_for_target(self, dat_ref, target_state, labels=None):
        """
        Returns all the data with a certain target state. If no labels are
        given, the global ones are taken. dat_ref is a list of indices belonging
        the data matrix of the root node.
        """

        # data?
        if len(dat_ref) == 0:
            return None

        # useful variables
        root = self.root()
        N = len(root.labels)

        # which labels to use?
        if labels is None:
            labels = root.labels

        # select data for target state
        data_refs = []
        for ref_i in dat_ref:
            if ref_i >= N-1:
                continue
            if labels[ref_i+1] == target_state:
                data_refs.append(ref_i)

        # data?
        if len(data_refs) == 0:
            return None

        # fetch the actual data
        data_list = map(lambda i: root.data[i], data_refs)
        return np.vstack(data_list)


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


#    def _is_splitted(self):
#        if len(self.children) == 0:
#            return False
#        return True


    @classmethod
    def _entropy(cls, trans, normalized_entropy, ignore_empty_classes=False):
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
            if not ignore_empty_classes:
                assert trans_sum != 0
            if normalized_entropy:
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
        if normalized_entropy:
            entropy /= np.log2(K)

        assert(entropy >= 0)
        return entropy


    def entropy(self):
        """
        Calculates the (weighted) entropy for the root node.
        """
        if self.status == 'leaf':
            
            # return entropy of leaf node
            root = self.root()
            current_state = self.class_label()
            trans = root.transitions[current_state]
            entropy = self._entropy(trans=trans, normalized_entropy=root._normalized_entropy)
            
        else:
        
            # entropy of all leaves ...
            root = self.root()
            leaf_entropies = map(lambda t: WorldModelTree._entropy(trans=t, normalized_entropy=root._normalized_entropy), root.transitions)

            # (weighted) average
            if self._global_entropy == 'sum':
                entropy = np.sum(leaf_entropies)
            elif self._global_entropy == 'avg':
                entropy = np.average(leaf_entropies)
            elif self._global_entropy == 'weighted':
                weights = np.sum(self.root().transitions, axis=1)
                weights /= np.sum(weights)
                entropy = np.sum(weights * leaf_entropies)
            else:
                raise RuntimeError("Valid options for global_entropy are 'sum', 'avg' and 'weighted'.")

        return entropy


    @classmethod
    def _transition_entropy(cls, trans, normalized_entropy, global_entropy):
        """
        Calculates the entropy for a given transition matrix.
        """        
        K = trans.shape[0]
        row_entropies = np.zeros(K)
        
        #trans = np.array(trans) + 1*np.ones((K,K))

        # entropies for every row of matrix
        for i in range(K):
            row = trans[i]
            row_entropies[i] = cls._entropy(trans=row, normalized_entropy=normalized_entropy)

        # (weighted) average
        if global_entropy == 'sum':
            entropy = np.sum(row_entropies)
        elif global_entropy == 'avg':
            entropy = np.average(row_entropies)
        elif global_entropy == 'weighted':
            weights = np.sum(trans, axis=1)
            weights /= np.sum(weights)
            entropy = np.sum(weights * row_entropies)
        else:
            raise RuntimeError("Valid options for global_entropy are 'sum', 'avg' and 'weighted'.")

        return entropy
    
    
    def split(self):
        
        # recursion to leaves
        if self.status != 'leaf':
            for leaf in self.leaves():
                leaf.split()
            return
        
        # split this leaf node
        assert self.status == 'leaf'
        root = self.root()
        self._init_test()
        new_labels, new_dat_ref = self._relabel_data()
        root.transitions = self._splitted_transition_matrix(new_labels)
        root.labels = new_labels
        
        # create new leaves
        child0 = self.__class__(normalized_entropy = self._normalized_entropy, 
                                global_entropy = self._global_entropy,
                                parents = [self])
        child1 = self.__class__(normalized_entropy = self._normalized_entropy, 
                                global_entropy = self._global_entropy,
                                parents = [self])
        child0.dat_ref = new_dat_ref[0]
        child1.dat_ref = new_dat_ref[1]

        # create list of children
        self.children = []
        self.children.append(child0)
        self.children.append(child1)
        self.status = 'split'
        return
    
    
    def sleep(self, depth=1):
        """
        Create little splits and re-assemble them with less entropy.
        """
        # neural thunderstorm
        for _ in range(depth):
            self.split()
            print self.root().transitions
            print np.sum(self.root().transitions)
        
        for _ in range(1000):
        
            # chose two random states for merging    
            root = self.root()
            K = root.transitions.shape[0]
            if K < 2:
                break
            s1, s2 = random.sample(range(K), 2)
            if s1 > s2:
                s1, s2 = [s2, s1]
            
            # merge rows and columns
            merged_trans = np.array(root.transitions)
            merged_trans[s1,:] += merged_trans[s2,:]
            merged_trans = np.delete(merged_trans, s2, 0)  
            merged_trans[:,s1] += merged_trans[:,s2]
            merged_trans = np.delete(merged_trans, s2, 1)  
            
            new_entropy = self._transition_entropy(trans=merged_trans,     normalized_entropy=self._normalized_entropy, global_entropy=self._global_entropy)
            old_entropy = self._transition_entropy(trans=root.transitions, normalized_entropy=self._normalized_entropy, global_entropy=self._global_entropy)
    
            if new_entropy <= old_entropy:
    
                # merge labels again
                for i in range(len(root.labels)):
                    if root.labels[i] == s2:
                        root.labels[i] = s1
                    if root.labels[i] > s2:
                        root.labels[i] -= 1
        
                # merge transitions again
                root.transitions = merged_trans
                
                # merge data references
                leaves = root.leaves()
                parent1 = leaves[s1]            
                parent2 = leaves[s2]            
                child = self.__class__(normalized_entropy = self._normalized_entropy, 
                                       global_entropy = self._global_entropy,
                                       parents = [parent1, parent2])
                child.dat_ref = parent1.dat_ref + parent2.dat_ref
                parent1.dat_ref = []
                parent2.dat_ref = []
                parent1.children = [child]
                parent2.children = [child]
                parent1.status = 'merged'
                parent2.status = 'merged'
                print 'merged'
        
        
        return
        
        
    def _init_test(self):
        """
        Initializes the parameters that split the node in two halves.
        """
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

        
        

#    def split_complete(self):
#        """
#        Splits the tree until convergence.
#        """
#        while True:
#            if not self.split():
#                break
#        return



#class PCAWorldModelTree(WorldModelTree):
#    """
#    This decision tree creates new leaves simply by splitting each node along
#    its first principal component.
#    """
#
#    def __init__(self, normalized_entropy, global_entropy, split_entropy, parent=None):
#        super(PCAWorldModelTree, self).__init__(normalized_entropy = normalized_entropy, 
#                                                global_entropy = global_entropy, 
#                                                split_entropy = split_entropy, 
#                                                parent = parent)
#
#
#    def _init_test(self):
#        # calculate PCA of data
#        data = self.get_data()
#        self.pca = mdp.nodes.PCANode(output_dim=1)
#        self.pca.train(data)
#        self.pca.stop_training()
#        return
#
#
#    def _test(self, x):
#        """
#        Tests to which child the data point x belons
#        """
#        if x.ndim < 2:
#            x = np.array(x, ndmin=2)
#        index = self.pca.execute(x)[0,0]
#        index = np.sign(index) + 1
#        index = index // 2
#        return int(index)
#
#
#    def _get_test_parameters(self):
#        return self.pca
#
#
#    def _set_test_parameters(self, params):
#        self.pca = params




#class RandomWorldModelTree(WorldModelTree):
#
#    def __init__(self, normalized_entropy, global_entropy, split_entropy, parent=None):
#        super(RandomWorldModelTree, self).__init__(normalized_entropy = normalized_entropy, 
#                                                   global_entropy = global_entropy, 
#                                                   split_entropy = split_entropy, 
#                                                   parent = parent)
#
#
#    def _init_test(self):
#
#        # two random points for random decision surface
#        x0, x1 = random.sample(self.get_data(), 2)
#
#        # decision surface
#        w = x1 - x0
#        w /= np.linalg.norm(w)
#        m = (x0+x1)/2.
#
#        # init _test
#        self._decision_w = w
#        self._decision_m = m
#        return True
#
#
#    def _test(self, x):
#        y = self._decision_w.dot(x - self._decision_m)
#        y = np.sign(y) + 1
#        y = y // 2
#        return int(y)
#
#
#    def _get_test_parameters(self):
#        return self._decision_w, \
#               self._decision_m
#
#
#    def _set_test_parameters(self, params):
#        self._decision_w, self._decision_m = params


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

    #problems = [problemChain, problemDiamond, problemHoneycomb]
    problems = [problemHoneycomb]

    for p, problem in enumerate(problems):

        # create data
        n = 10000
        data = problem(n=n, seed=1)

        tree = WorldModelTree(normalized_entropy=False, global_entropy='weighted')
        tree.add_data(data)


        print tree.transitions
        tree.sleep(depth=6)
        


        n_trans = np.sum(tree.transitions)
        print 'final transitions:\n', tree.transitions
        print 'sum:', n_trans
        print 'final entropy:', tree.entropy()
        assert(n_trans == n-1)

        pyplot.subplot(1, len(problems), p+1)
        tree.plot_tree_data(show_plot=False)

    pyplot.show()
