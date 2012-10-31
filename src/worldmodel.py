import mdp

from matplotlib.pyplot import *
import math
import numpy as np
import random


class WorldModelTree(object):

    symbols = ['o', '^', 'd', 's', '*']

    def __init__(self, normalized_entropy, global_entropy, parent=None):
        self.parent = parent
        self.children = []
        self.data = None     # global data storage in root node
        self.dat_ref = []    # indices of data belonging to this node
        self.transitions = None
        self._normalized_entropy = normalized_entropy
        self._global_entropy = global_entropy


    def classify(self, x):
        """
        Takes a single vector and returns an integer class label.
        """

        if x.ndim > 1:

            N = x.shape[0]
            labels = np.zeros(N)
            for i in range(N):
                labels[i] = self.classify(x[i])
            return labels

        else:

            if self._is_splitted():
                child_index = int(self._test(x))
                class_label = self.children[child_index].classify(x)
                return class_label
            else:
                all_leafes = self.root().leaves()
                class_label = all_leafes.index(self)
                return class_label


    def split(self):
        """
        Splits the data along the first principal component. If the node is
        splitted, all subnodes are splitted.
        Returns the number of performed splits.
        """

        # recursion: step down to the leafes
        if self._is_splitted():
            n_splits = 0
            children = list(self.children)
            # TODO shuffle
            #random.shuffle(children)
            for child in children:
                n_splits += child.split()
            return n_splits

        # check wether there's enough data
        # TODO parameterize
        if len(self.dat_ref) < 10:
            return 0

        # find best split ...
        best_entropy = float('Inf')
        best_test = None
        for _ in range(100):

            # initialize decision boundary of _test() method
            if self._init_test() == False:
                continue

            # calculate transitions matrix for new labels
            new_labels, new_dat_ref = self._relabel_data()
            new_trans = self._splitted_transition_matrix(new_labels)

            # remember best split
            new_entropy = self._transition_entropy(trans_matrix = new_trans)

            # correct by entropy of the split itself
            split_sizes = np.zeros(2)
            split_sizes[0] = len(new_dat_ref[0])
            split_sizes[1] = len(new_dat_ref[1])
            new_entropy /= self._entropy(split_sizes)

            if new_entropy < best_entropy:
                best_entropy = new_entropy
                best_labels = new_labels
                best_refs = new_dat_ref
                best_trans = new_trans
                best_test = self._get_test_parameters()

        # was there a good test at all?
        root = self.root()
        if best_entropy >= root.entropy():
            return 0

        # remeber the calculated entropy
        self._set_test_parameters(best_test)
        root.labels = best_labels

        # create new leafes
        child0 = self.__class__(normalized_entropy=self._normalized_entropy, global_entropy=self._global_entropy, parent=self)
        child1 = self.__class__(normalized_entropy=self._normalized_entropy, global_entropy=self._global_entropy, parent=self)
        child0.dat_ref = best_refs[0]
        child1.dat_ref = best_refs[1]

        # create list of children
        # (makes split official!)
        self.children = []
        self.children.append(child0)
        self.children.append(child1)

        # copy transitions to children
        for c, leaf in enumerate(root.leaves()):
            leaf.transitions = best_trans[c]
        self.transitions = None

        #assert(best_entropy == root.entropy())
        return 1


    def _relabel_data(self):
        """
        Returns new labels and splitted data refs according to the _test()
        method of a leaf node. So, _test() has to be initialized before but the
        node not finally splitted yet.
        """

        # some useful variables
        root = self.root()
        if self in root.leaves():
            current_state = root.leaves().index(self) # current class index
        else:
            raise RuntimeError('Error: _relabel_data() has to be called from a leaf node.')

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

        return new_labels, new_dat_ref


    def _splitted_transition_matrix(self, new_labels):
        """
        Calculates a new transition matrix with the current state splitted.
        """
        if self._is_splitted():
            print 'Warning: Node already splitted!'
            return None

        # helpful variable
        root = self.root()
        current_state = root.leaves().index(self)
        N = len(new_labels)

        # new transition matrix
        new_trans = self._transition_matrix()
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

        return new_trans


    def add_data(self, x):
        """
        Adds a matrix x of new observations to the node. The data is
        interpreted as one observation following the previous one. This is
        important to calculate the transition probabilites.
        If there has been data before, the new data is appended.
        """

        # add data to root node only
        if self.parent is not None:
            self.root().add_data(x)
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
            first_source = first_i - 1
            self.data = np.vstack([self.data, x])
            self.labels = np.vstack([self.labels, labels])

        # add references to data in all leafes
        all_leaves = self.leaves()
        N = self.data.shape[0]
        for i in range(first_data, N):
            label = self.labels[i]
            leaf  = all_leaves[label]
            leaf.dat_ref.append(i)

        # make sure, every leaf has a vector for transitions
        if first_data == 0:
            C = len(all_leaves) # number of states
            for leaf in all_leaves:
                leaf.transitions = np.zeros(C)

        # update transitions in every node
        for i in range(first_source, N-1):
            source_state = labels[i]
            source_leaf  = all_leaves[source_state]
            target_state = labels[i+1]
            source_leaf.transitions[target_state] += 1

        return
    

    def leaves(self):
        """
        Returns a list of all leaves belonging to the node.
        """
        if self._is_splitted():
            children = []
            for child in self.children:
                children += child.leaves()
            return children
        else:
            # node not splitted
            return [self]


    def root(self):
        """
        Returns the root node of the whole tree.
        """
        if self.parent is None:
            return self
        else:
            return self.parent.root()


    def _init_test(self):
        """
        This method initializes what-ever is necessary for the _test() method to
        work. This means, after _init_test() a (temporary) decision boundary
        exists for the node.
        """
        raise NotImplementedError('_init_test() has to be implemented by sub-class.')


    def _test(self, x):
        """
        Tests to which child the data point x belons
        """
        raise NotImplementedError('_test() has to be implemented by sub-class.')


    def plot_tree_data(self, show_plot=True):
        """
        Plots all the data that is stored in the tree.
        """

        # list of data for the different classes
        data_list = []
        all_leaves = self.leaves()
        for leaf in all_leaves:
            data = leaf.get_data()
            if data is not None:
                data_list.append(data)

        # plot
        colormap = cm.prism
        gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.98, 7)])
        for i, data in enumerate(data_list):
            plot(data[:,0], data[:,1], self.symbols[i%len(self.symbols)])
            
        if show_plot:
            show()
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


    def get_data(self):
        """
        Returns the data belonging to the node. If the node isn't a leaf, the
        data of sub-nodes is returned.
        """

        if self._is_splitted():

            data_list = []
            data_list.append(self.children[0].get_data())
            data_list.append(self.children[1].get_data())

            # maybe the leaf nodes didn't have data. then remove.
            data_list.remove(data_list.index(None))
            data_list.remove(data_list.index(None))

            if len(data_list) == 0:
                return None
            else:
                return np.vstack(data_list)

        else:

            # data available?
            if len(self.dat_ref) == 0:
                return None

            # fetch the actual data
            root = self.root()
            data_list = map(lambda i: root.data[i], self.dat_ref)
            return np.vstack(data_list)


    def _is_splitted(self):
        if len(self.children) == 0:
            return False
        return True


    def _transition_matrix(self):
        """
        Returns a matrix with all transitions of the tree. Source state in row
        and target in column.
        """
        all_leaves = self.root().leaves()
        C = len(all_leaves)
        trans = np.zeros((C, C))
        for c, leaf in enumerate(all_leaves):
            trans[c,:] = leaf.transitions
        assert(np.sum(trans) == self.root().data.shape[0]-1)
        return trans


    def _entropy(self, trans=None, warning=True):
        """
        Calculates the (normalized) entropy over the target state distribution
        for a leaf or a given list of transitions.
        """
        # which transitions to use?
        if trans is None:
            trans = self.transitions

        # negativ values?
        if True in list(trans < -0.):
            raise ValueError('Negative values in transition vector!')

        # useful variables
        trans_sum = np.sum(trans)
        K = len(trans)

        # only one class?
        if K <= 1:
            return 1.0

        # empty class?
        if trans_sum == 0:
            if warning:
                print 'Warning: empty class! Should not happen.'
            return 1.0

        # the actual calculation
        probs = np.array(trans, dtype=np.float64) / trans_sum
        log_probs = np.zeros_like(probs)
        log_probs[probs > 0.] = np.log2( probs[probs > 0.] )
        entropy = -np.sum(probs * log_probs)

        # normalization?
        assert(entropy <= np.log2(K))
        if self._normalized_entropy:
            entropy /= np.log2(K)

        assert(entropy >= 0)
        return entropy


    def entropy(self):
        """
        Calculates the (weighted) entropy of the target distribution for a node,
        e.g., the root node.
        """
        if self._is_splitted():

            # entropy of all leaves ...
            leaves = self.leaves()
            leaf_entropies = map(lambda leaf: leaf._entropy(), leaves)

            # (weighted) average
            if self._global_entropy == 'sum':
                entropy = np.sum(leaf_entropies)
            elif self._global_entropy == 'avg':
                entropy = np.average(leaf_entropies)
            elif self._global_entropy == 'weighted':
                weights = map(lambda l: np.sum(l.transitions), leaves)
                weights /= np.sum(weights)
                entropy = np.sum(weights * leaf_entropies)
            else:
                raise RuntimeError("Valid options for global_entropy are 'sum', 'avg' and 'weighted'.")

        else:
            # return entropy of leaf node
            entropy = self._entropy()
            #entropy /= self._state_entropy(normalized_entropy=normalized_entropy)

        return entropy


    def _transition_entropy(self, trans_matrix):
        """
        Calculates the entropy for the transition matrix.
        """
        K = trans_matrix.shape[0]
        row_entropies = np.zeros(K)

        # entropies for every row of matrix
        for i in range(K):
            row = trans_matrix[i]
            row_entropies[i] = self._entropy(trans=row)

        # (weighted) average
        if self._global_entropy == 'sum':
            entropy = np.sum(row_entropies)
        elif self._global_entropy == 'avg':
            entropy = np.average(row_entropies)
        elif self._global_entropy == 'weighted':
            weights = np.sum(trans_matrix, axis=1)
            weights /= np.sum(weights)
            entropy = np.sum(weights * row_entropies)
        else:
            raise RuntimeError("Valid options for global_entropy are 'sum', 'avg' and 'weighted'.")

        #entropy /= self._state_entropy(normalized_entropy=normalized_entropy)
        return entropy


    def _state_entropy(self, trans=None, normalized_entropy=True):
        """
        Calculates the entropy of state space partitioning itself. It will be
        maximal if all states are equally large. Optionally, a transition matrix
        can be given.
        """

        if trans is None:
            trans = self._transition_matrix()

        class_sizes = np.sum(trans, axis=1)
        entropy = self._entropy(trans=class_sizes, normalized_entropy=normalized_entropy)
        return entropy

    
    def _class_index(self):
        """
        Returns the class index/label of the node. Returns None if node is not
        a leaf.
        """
        leaves = self.root().leaves()
        if self in leaves:
            return leaves.index(self)
        else:
            return None



class PCAWorldModelTree(WorldModelTree):
    """
    This decision tree creates new leaves simply by splitting each node along
    its first principal component.
    """

    def __init__(self, parent=None, **kwargs):
        super(PCAWorldModelTree, self).__init__(parent, **kwargs)


    def _init_test(self):
        # calculate PCA of data
        data = self.get_data()
        self.pca = mdp.nodes.PCANode(output_dim=1)
        self.pca.train(data)
        self.pca.stop_training()
        return


    def _test(self, x):
        """
        Tests to which child the data point x belons
        """
        if x.ndim < 2:
            x = np.array(x, ndmin=2)
        index = self.pca.execute(x)[0,0]
        index = np.sign(index) + 1
        index = index // 2
        return int(index)


    def _get_test_parameters(self):
        return self.pca


    def _set_test_parameters(self, params):
        self.pca = params




class ClusterWorldModelTree(WorldModelTree):

    def __init__(self, parent=None, **kwargs):
        super(ClusterWorldModelTree, self).__init__(parent, **kwargs)


    def _pca(self, data):
        pca = mdp.nodes.PCANode(output_dim=1)
        pca.train(data)
        pca.stop_training()
        w = pca.v[:,0]
        theta = w.dot(pca.avg[0])
        return w, theta


    def _init_test(self):

        data = self.get_data()
        self._boundary_w, self._boundary_theta = self._pca(data)

        new_labels, new_dat_ref = self._relabel_data()

        means = []
        covs = []
        weights = []

        for child_index in [0,1]:

            # calculate mean and covariance of each target state
            root = self.root()
            all_leaves = root.leaves()
            K = len(all_leaves)
            for target_state in range(K+1):
                dat = self._get_data_for_target(new_dat_ref[child_index], target_state=target_state, labels=new_labels)
                if dat is None or dat.shape[0] <= 1:
                    continue
                covariance_matrix = mdp.utils.CovarianceMatrix()
                covariance_matrix.update(dat)
                C, M, N = covariance_matrix.fix()
                means.append(M)
                covs.append(C)
                weights.append(N)

        # at least two target states from here?
        if len(means) <= 1:
            return False

        # cluster data until only two clusters remain
        while len(means) > 2:
            # find smallest mahalanobis distance between class means
            min_dif = float('Inf')
            min_i = None
            min_j = None
            K = len(means)  # number of clusters
            for i in range(K):
                for j in range(i+1, K):
                    dif_m = means[i] - means[j]
                    S = ( weights[i]*covs[i] + weights[j]*covs[j] ) / (weights[i] + weights[j])
                    Sinv = np.linalg.inv(S)
                    dif = np.sqrt(dif_m.T.dot(Sinv.dot(dif_m))) # mahalanobis distance
                    if dif < min_dif:
                        min_i = i
                        min_j = j
            # merge the two nearest states
            means[min_i] = (weights[min_i]*means[min_i] + weights[min_j]*means[min_j]) / (weights[min_i] + weights[min_j])
            covs[min_i] = (weights[min_i]*covs[min_i] + weights[min_j]*covs[min_j]) / (weights[min_i] + weights[min_j])
            means.pop(min_j)
            covs.pop(min_j)
            weights.pop(min_j)

        # calculate Fisher LDA for the two remaining clusters
        dif_m = means[1] - means[0]
        S = ( covs[0] + covs[1] )# / 2
        Sinv = np.linalg.inv(S)
        w = Sinv.dot(dif_m)
        w = w / np.linalg.norm(w)
        self._boundary_w = w

        # set threshold at intersect between the two gaussians
        mu0 = w.dot(means[0])
        mu1 = w.dot(means[1])
        sigma0 = w.dot(covs[0]).dot(w)
        sigma1 = w.dot(covs[1]).dot(w)

        # solve quadratic equation for intersect point
        a = sigma1**2 - sigma0**2
        b = 2*(mu1*sigma0**2 - mu0*sigma1**2)
        c = mu0**2 * sigma1**2 - mu1**2 * sigma0**2 - 2 * sigma0**2 * sigma1**2 * np.log(sigma1/sigma0)
        x0 = ( -b + np.sqrt( b**2 - 4*a*c ) ) / ( 2*a )
        x1 = ( -b - np.sqrt( b**2 - 4*a*c ) ) / ( 2*a )

        # chose solution between means
        m0 = min(mu0, mu1)
        m1 = max(mu0, mu1)
        if x0 >= m0 and mu0 <= m1:
            self._boundary_theta = x0
        else:
            self._boundary_theta = x1

        return True


    def _test(self, x):
        y = self._boundary_w.dot(x)
        y = np.sign(y - self._boundary_theta) + 1
        y = y // 2
        return int(y)



class RandomWorldModelTree(WorldModelTree):

    def __init__(self, normalized_entropy, global_entropy, parent=None, **kwargs):
        super(RandomWorldModelTree, self).__init__(normalized_entropy, global_entropy, parent, **kwargs)


    def _init_test(self):

        # two random points for random decision surface
        x0, x1 = random.sample(self.get_data(), 2)

        # decision surface
        w = x1 - x0
        w /= np.linalg.norm(w)
        m = (x0+x1)/2.

        # init _test
        self._decision_w = w
        self._decision_m = m
        return True


    def _test(self, x):
        y = self._decision_w.dot(x - self._decision_m)
        y = np.sign(y) + 1
        y = y // 2
        return int(y)


    def _get_test_parameters(self):
        return self._decision_w, \
               self._decision_m


    def _set_test_parameters(self, params):
        self._decision_w, self._decision_m = params


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
        n = 1000
        data = problem(n=n)

        tree = RandomWorldModelTree(normalized_entropy=True, global_entropy='weighted')
        tree.add_data(data)

        while True:
            n_splits = tree.split()
            if n_splits:
                print 'performed {0} splits.'.format(n_splits)
                print tree._transition_matrix()
                print 'new entropy: {0}'.format(tree.entropy())
            else:
                break

        sum = np.sum(tree._transition_matrix())
        print 'final transitions:\n', tree._transition_matrix()
        print 'sum:', sum
        print 'final entropy:', tree.entropy()
        assert(sum == n-1)

        subplot(2, 2, p+1)
        tree.plot_tree_data(show_plot=False)

    show()
