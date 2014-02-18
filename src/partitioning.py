import numpy as np
import weakref

from matplotlib import pyplot

import split_params


class Partitioning(object):
    
    def __init__(self, model, active_action):#, tree, labels, transitions):
        self.model = weakref.proxy(model)
        self.active_action = active_action
        #self.tree = tree
        #self.labels = labels
        #self.transitions = transitions
        
        N = model.get_number_of_samples()
        self.labels = np.zeros(N, dtype=int)
        self.tree = self.model._tree_class(partitioning=self)
        self.transitions = {}
        for action in self.model.get_known_actions():
            self.transitions[action] = np.ones((1, 1), dtype=int) * np.count_nonzero(self.model.actions == action)


    def classify(self, data):
        """
        Returns the state(s) that the data belongs to according to the current 
        model.
        """
        return self.tree.classify(data)
    

    def get_merged_transition_matrices(self):
        """
        Merges all transition matrices.
        """
        
        K = self.tree.get_number_of_leaves()
        P = np.zeros((K, K), dtype=int)
        
        for a in self.model.get_known_actions():
            P += self.transitions[a]

        assert np.sum(P) == self.model.get_number_of_samples() - 1
        return P


    def calc_best_split(self):
        """
        Calculates the gain for each state and returns a split-object for the
        best one
        """
        
        if self.model.data is None:
            return None
                
        best_split = None

        for leaf in self.tree.get_leaves():
            
            if leaf._cached_split_params is None:
                leaf._cached_split_params = split_params.SplitParamsLocalGain(node=leaf)
                
            split = leaf._cached_split_params
            split.update()
                
            # TODO: find a way to mark split as invalid (for instance test_params failed)
            if split is not None:
                if best_split is None or split.get_gain() >= best_split.get_gain():
                    best_split = split
                
        return best_split


    def plot_data_colored_for_state(self, show_plot=True):
        """
        Plots all the data that is stored in the tree with color and shape
        according to the learned state.
        """
        
        # fancy shapes and colors
        symbols = ['o', '^', 'd', 's', '*']
        colormap = pyplot.cm.get_cmap('prism')
        pyplot.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.98, 7)])
        
        # data for the different classes
        leaves = self.tree.get_leaves()
        for i, leaf in enumerate(leaves):
            data = leaf.get_data()
            if data is not None:
                pyplot.plot(data[:,0], data[:,1], symbols[i%len(symbols)])
                
        if show_plot:
            pyplot.show()
            
        return


    def plot_state_borders(self, show_plot=True, range_x=None, range_y=None, resolution=100):
        """
        Shows a contour plot of the learned state borders (2D). 
        """
        
        data = self.model.data
        K = len(self.tree.get_leaves())
        
        if range_x is None:
            range_x = [np.min(data[:,0]), np.max(data[:,0])]
            
        if range_y is None:
            range_y = [np.min(data[:,1]), np.max(data[:,1])]
            
        x = np.linspace(range_x[0], range_x[1], resolution)
        y = np.linspace(range_y[0], range_y[1], resolution)
        X, Y = np.meshgrid(x, y)
        v_classify = np.vectorize(lambda x, y: self.classify(np.array([[x,y]])))
        Z = v_classify(X, Y)
        pyplot.contour(X, Y, Z, levels = range(-1, K), colors='b', linewidths=1)
        
        if show_plot:
            pyplot.show()
        return
    
    
    def plot_transitions(self):
        for action in self.model.get_known_actions():
            if action == self.active_action:
                color = 'r'
            else:
                color = 'b'
            refs_1 = self.tree.get_transition_refs_for_action(action=action)
            refs_2 = refs_1 + 1
            data_1 = self.model.get_data_for_refs(refs=refs_1)
            data_2 = self.model.get_data_for_refs(refs=refs_2)
            for i in range(len(refs_1)):
                pyplot.plot([data_1[i,0], data_2[i,0]], [data_1[i,1], data_2[i,1]], '-', color=color)


if __name__ == '__main__':
    pass
