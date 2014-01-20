import numpy as np
import weakref

import tree_structure


class WorldmodelTree(tree_structure.Tree):
    
    def __init__(self, model):
        super(WorldmodelTree, self).__init__()
        
        # important references
        if type(model) in weakref.ProxyTypes:
            self.model = model
        else:
            self.model = weakref.proxy(model)
            
        self.data_refs = np.empty(0, dtype=int) # indices of data belonging to this node
        self._split_params = None
        return
        
        
    def _calc_test_params(self, active_action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves and returns
        them.
        """
        raise NotImplementedError("Use subclass like WorldmodelSpectral instead.")


    def _test(self, x, params):
        """
        Tests to which child the data point x belongs. Parameters are the ones
        calculated by calc_test_params().
        """
        raise NotImplementedError("Use subclass like WorldmodelSpectral instead.")
    

    def classify(self, x):
        """
        Returns the state that x belongs to according to the current model. If
        x is a matrix, a list is returned containing a integer state for every
        row.
        
        TODO: vectorize function
        """
        
        assert self.get_root() is self
        assert x.ndim == 2
        
        N, _ = x.shape
        labels = np.zeros(N, dtype=int)
        leaves = self.get_leaves()
        node_indices = dict(zip(leaves, [leaf.get_leaf_index() for leaf in leaves]))
        
        for i, dat in enumerate(x):
            
            node = self
            while not node.is_leaf():
                child_index = node._test(dat, params=self._test_params)
                node = self._children[child_index]
                
            labels[i] = node_indices[node]

        return labels
            
            
    def get_number_of_samples(self):
        return len(self.data_refs)
            
            
    def get_data(self):
        """
        Returns the data belonging to the node. If the node isn't a leaf, the
        data of sub-nodes is returned.
        """
        
        model = self.model
        dat_refs = sorted(self.get_data_refs())
        N = len(dat_refs)
        D = model.get_input_dim()
        
        if N == 0:
            return None
        
        data = np.empty((N, D))
        for i, ref in enumerate(dat_refs):
            data[i] = model.data[ref]
            
        return data 
    
    
    def split(self, split_params):
        """
        Applies a split.
        """
        
        assert self.is_leaf()
        self._split_params = split_params
        
        # copy labels and transitions to model
        model = self.model
        action = split_params._action
        leaf_index = self.get_leaf_index()
        assert len(self.data_refs) == np.count_nonzero(model.partitionings[action].labels == leaf_index)
        model.partitionings[action] = model.partitionings[action]._replace(labels = split_params.get_new_labels(), transitions = split_params.get_new_trans())
        
        # copy new references to children
        new_dat_refs = split_params.get_new_dat_refs()
        assert len(self.data_refs) == len(new_dat_refs[0]) + len(new_dat_refs[1])
        child_1, child_2 = super(WorldmodelTree, self).split(model=model)
        child_1.data_refs = new_dat_refs[0]
        child_2.data_refs = new_dat_refs[1]
        
        assert len(child_1.data_refs) == np.count_nonzero(model.partitionings[action].labels == leaf_index)
        assert len(child_2.data_refs) == np.count_nonzero(model.partitionings[action].labels == leaf_index+1)
        
        #assert False not in [model.partitionings[action].labels[ref]==leaf_index for ref in child_1.data_refs]
        #assert False not in [model.partitionings[action].labels[ref]==leaf_index+1 for ref in child_2.data_refs]
        
        # free some memory
        self.data_refs = None
        return child_1, child_2
    

    def get_data_refs(self):
        """
        Returns a list of data references (i.e. indices for root.data) 
        belonging to the node. If the node isn't a leaf, the data of sub-nodes 
        is returned.
        """

        if self.is_leaf():
            return self.data_refs

        # else        
        data_refs = np.empty(0, dtype=int)
        for child in self._children:
            data_refs = np.hstack([data_refs, child.get_data_refs()])
        
        data_refs.sort()
        return data_refs


    def get_transition_refs(self, heading_in=False, inside=True, heading_out=False):
        """
        Finds all transitions that start, end or happen strictly inside the 
        node. The result is given as two lists of references. One for the start 
        and one for the end of the transition.
        
        TODO: check!
        """
        
        refs_1 = self.get_data_refs()
        refs_0 = refs_1 - 1
        N = self.model.get_number_of_samples()
        
        if heading_in:
            assert False
            # [ref-1 for ref in refs if (ref-1 not in refs) and (ref-1 >= 0)]
            refs_array_in = np.setdiff1d(refs_0, refs_1, assume_unique=True)
            refs_array_in.difference_update([-1])
            assert set(refs_array_in) == set([ref-1 for ref in refs_1 if (ref-1 not in refs_1) and (ref-1 >= 0)])
            
        if inside:
            # [ref for ref in refs if (ref+1 in refs)]
            mask = np.in1d(refs_1, refs_0, assume_unique=True)
            refs_array_inside = refs_1[mask]
            #assert set(refs_array_inside) == set([ref for ref in refs if (ref+1 in refs)])

        if heading_out:
            # [ref for ref in refs if (ref+1 not in refs) and (ref+1 < N)]
            refs_array_out = np.setdiff1d(refs_1, refs_0, assume_unique=True)
            refs_array_out.difference_udate([N-1])
            assert set(refs_array_out) == set([ref for ref in refs_1 if (ref+1 not in refs_1) and (ref+1 < N)])
            
        refs_1 = refs_array_inside
        refs_2 = refs_array_inside + 1
        return [refs_1, refs_2]
    
    
    def _reached_number_of_active_and_inactive_samples(self, number, active_action):
        """
        Calculates whether for the active _action and all other actions a certain
        number of samples is reached.
        """
        refs_1, _ = self.get_transition_refs(heading_in=False, inside=True, heading_out=False)
        actions = [self.model.actions[ref] for ref in refs_1]
        
        n_actions = len(self.model.get_known_actions())
        n_samples = len(refs_1)
        n_samples_active = actions.count(active_action)
        n_samples_inactive = n_samples - n_samples_active
        
        return (n_samples_active >= number) and (n_actions == 1 or n_samples_inactive >= number)
        
        

if __name__ == '__main__':
    pass
