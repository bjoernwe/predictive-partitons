import collections
import math
import numpy as np
import random

from matplotlib import pyplot

import mdp


Stats = collections.namedtuple('Stats', ['n_states', 
                                         'norm', 
                                         'entropy', 
                                         'entropy_normalized', 
                                         'entropy_per_norm', 
                                         'mutual_information', 
                                         'normalized_mutual_information', 
                                         'kl_divergence_rate'
                                         ])

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
            #if not ignore_empty_classes:
            #    assert trans_sum != 0
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
    def _matrix_entropy(cls, transitions, normalized_entropy, global_entropy, only_out=False):
        """
        Calculates the entropy for a given transition matrix.
        """        
        K = transitions.shape[0]
        row_entropies = np.zeros(K)

        if only_out:
            weights = np.sum(transitions, axis=1)
            probs = np.array(transitions, dtype=np.float) / weights[:,np.newaxis]
            np.fill_diagonal(probs, 0)
            log_probs = np.zeros_like(probs)
            log_probs[probs > 0.] = np.log2( probs[probs > 0.] )
            row_entropies = -np.sum(probs * log_probs, axis=1)
            if normalized_entropy:
                row_entropies /= np.log2(K)
        else:
            for i in range(K):
                row = transitions[i]
                row_entropies[i] = cls._entropy(trans=row, normalized_entropy=normalized_entropy)

        # (weighted) average
        if global_entropy == 'sum':
            entropy = np.sum(row_entropies)
        elif global_entropy == 'avg':
            entropy = np.average(row_entropies)
        elif global_entropy == 'weighted':
            weights = np.sum(transitions, axis=1)
            weights /= np.sum(weights)
            entropy = np.sum(weights * row_entropies)
        else:
            raise RuntimeError("Valid options for global_entropy are 'sum', 'avg' and 'weighted'.")

        return entropy
    
    
    @classmethod
    def _transition_min_entropy(cls, trans, normalized_entropy, global_entropy, only_out=False):
        """
        Calculates the min-entropy for a given transition matrix.
        """        
        K = trans.shape[0]
        row_entropies = np.zeros(K)
        
        trans = np.array(trans)
        if only_out:
            np.fill_diagonal(trans, 0)

        # entropies for every row of matrix
        for i in range(K):
            row = trans[i]
            row /= np.sum(row)
            row_entropies[i] = -np.log2(np.max(row))

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
    
    
    @classmethod
    def _transition_min_entropy_auto(cls, trans, global_entropy):
        """
        Calculates the min-entropy of all the diagonal entries of a transition matrix.
        """        
        K = trans.shape[0]
        row_entropies = np.zeros(K)
        
        trans = np.array(trans) + 1

        # entropies for every row of matrix
        for i in range(K):
            row = trans[i]
            row /= np.sum(row)
            row_entropies[i] = -np.log2(row[i])

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
        if len(self.dat_ref) < self._min_class_size:
            return
        
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
    
    
    def _entropy_of_merged_states(self, matrix, s1, s2):
        
        merged_matrix = np.array(matrix)
        
        merged_row = (matrix[s1,:] + matrix[s2,:]) / 2.
        merged_matrix[s1] = merged_row
        merged_matrix[s2] = merged_row

        merged_col = (matrix[:,s1] + matrix[:,s2]) / 2.
        merged_matrix[:,s1] = merged_col
        merged_matrix[:,s2] = merged_col
        
        return self._matrix_entropy(transitions=merged_matrix, normalized_entropy=False, global_entropy='weighted', only_out=False)
    
    
    def sleep(self, depth=1):
        """
        Create little splits and re-assemble them with less entropy.
        """
        # split severat times ("neural thunderstorm")
        root = self.root()
        for _ in range(depth):
            self.split()
            print root.transitions
            #print np.sum(root.transitions)
            
        root.stats.append(self._calc_stats(transitions_large=root.transitions, transitions_small=root.transitions))
        plotted_yet = False
            
        # merge again ...
        for r in range(150):
        
            # useful variables    
            root = self.root()
            K = root.transitions.shape[0]
            if K < 2:
                break
            
            # find best merge ...
            #best_entropy = float('inf')
            best_s1 = None
            best_s2 = None
            best_trans = None
            best_labels = None
            best_stats = root.stats[-1]
            best_diff = float("inf")
            
            if r%100 == 0:    
                print 'r:', r
                
            best_norm = 0
                
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
                
                new_stats = self._calc_stats(transitions_large=root.transitions, transitions_small=merged_trans)
                
                #diff = abs(root.stats[-1].norm - new_stats.norm)
                #diff = abs(root.stats[-1].entropy - new_stats.entropy)
                diff = abs(root.stats[-1].mutual_information - new_stats.mutual_information)
                
                if (True and
                    #new_stats.entropy_out_quot > best_quotient and
                    #new_stats.entropy < root.stats[-1].entropy and
                    diff < best_diff and
                    #new_stats.entropy_per_norm < best_stats.entropy_per_norm and
                    #new_stats.normalized_entropy_per_norm < best_stats.normalized_entropy_per_norm and
                    #new_stats.mutual_information > best_stats.mutual_information and
                    #new_stats.norm > best_norm and
                    True):
                    
                    #print 'best entropy:', new_stats.entropy
                    print 'best diff:', diff
                    
                    # merge labels again
                    new_labels = list(root.labels)
                    for i in range(len(new_labels)):
                        if new_labels[i] == s2:
                            new_labels[i] = s1
                        if new_labels[i] > s2:
                            new_labels[i] -= 1

                    #best_entropy = new_stats.entropy
                    best_s1 = s1
                    best_s2 = s2
                    best_trans = merged_trans#np.array(merged_trans)
                    best_labels = new_labels#list(new_labels)
                    best_stats = new_stats#Stats(*new_stats)
                    best_norm = new_stats.norm
                    best_diff = diff
                    
                    merged_trans = None
                    new_labels = None
                    new_stats = None

            if best_s1 is None:
                continue
            print '***'
            
            if not plotted_yet:
                if best_stats.normalized_mutual_information < root.stats[-1].normalized_mutual_information:
                    pyplot.subplot(2,3,2)
                    self.plot_tree_data(show_plot=False)
                    plotted_yet = True
            
            # merge transitions again
            root.transitions = best_trans
            root.labels = best_labels
            root.stats.append(best_stats)
            
            # merge data references
            leaves = root.leaves()
            parent1 = leaves[best_s1]            
            parent2 = leaves[best_s2]            
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
            print root.transitions
            print np.sum(root.transitions)
                
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
    
    
    def _calc_stats(self, transitions_large, transitions_small):
        """
        Calculates statistics for a given transition matrix.
        """
        
        P = transitions_large
        Q = transitions_small
        
        q_entropy = self._matrix_entropy(transitions=Q, normalized_entropy=False, global_entropy=self._global_entropy, only_out=False)
        q_entropy_normalized = self._matrix_entropy(transitions=Q, normalized_entropy=True, global_entropy=self._global_entropy, only_out=False)
        
        # norm of Q
        q_weights = np.sum(Q, axis=1)
        q_probs = Q / q_weights[:,np.newaxis]
        q_mu = q_weights / np.sum(q_weights)
        q_K = Q.shape[0]
        q_norm = np.sum( ( q_probs**2 * q_mu[:,np.newaxis] ) / q_mu[np.newaxis,:] )
        
        q_entropy_per_norm = q_entropy / q_norm
        
        # mutual information
        q_entropy_mu = self._entropy(trans=q_mu, normalized_entropy=False)
        q_mutual_information = q_entropy_mu - q_entropy
        q_normalized_mutual_information = q_mutual_information / np.log2(q_K)

        p_entropy = self._matrix_entropy(transitions=P, normalized_entropy=False, global_entropy=self._global_entropy, only_out=False)
        p_weights = np.sum(P, axis=1)
        p_mu = p_weights / np.sum(p_weights)
        p_entropy_mu = self._entropy(trans=p_mu, normalized_entropy=False)
        p_mutual_information = p_entropy_mu - p_entropy
        
        kl_divergence_rate = p_mutual_information - q_mutual_information
        
        
        stats = Stats(n_states = q_K, 
                      entropy = q_entropy, 
                      entropy_normalized = q_entropy_normalized, 
                      norm = q_norm, 
                      entropy_per_norm = q_entropy_per_norm,
                      mutual_information = q_mutual_information,
                      normalized_mutual_information = q_normalized_mutual_information,
                      kl_divergence_rate = kl_divergence_rate
                      )
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

    #problems = [problemChain, problemDiamond, problemHoneycomb]
    problems = [problemHoneycomb]

    for p, problem in enumerate(problems):

        # create data
        n = 10000
        data = problem(n=n, seed=1)

        tree = WorldModelTree(normalized_entropy=True, global_entropy='weighted')
        tree.add_data(data)


        print tree.transitions
        tree.sleep(depth=6)
        


        n_trans = np.sum(tree.transitions)
        entropy = tree.entropy()
        print 'final transitions:\n', tree.transitions
        print 'sum:', n_trans
        print 'final entropy:', entropy
        assert(n_trans == n-1)

        pyplot.subplot(2, 3, 3)
        tree.plot_tree_data(show_plot=False)

    pyplot.subplot(2, 1, 2)
    pyplot.plot(np.vstack(tree.stats)[:,1:])
    #pyplot.plot([-30, -2], [entropy, entropy], '--', c='gray')
    #pyplot.plot([-7, -7], [.1, 1], '--', c='gray')
    pyplot.legend(list(tree.stats[0]._fields)[1:], loc=3)# + ['true entropy', 'true #classes'])
    pyplot.show()
