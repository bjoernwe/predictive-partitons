import numpy as np
import weakref

import entropy_utils
import tree_structure


class WorldmodelTree(tree_structure.Tree):
    
    def __init__(self, model):
        super(WorldmodelTree, self).__init__()
        
        # important references
        if type(model) == weakref.ProxyType:
            self._model = model
        else:
            self._model = weakref.proxy(model)
        self._dat_ref = []   # indices of data belonging to this node
        self._split_params = None
        
        
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
    

    def _classify(self, x):
        """
        Returns the state that x belongs to according to the current model. If
        x is a matrix, a list is returned containing a integer state for every
        row.
        """

        # is x a matrix?
        if x.ndim > 1:

            # classify every point
            labels = [self._classify(row) for row in x]
            return labels

        else:

            if self.is_leaf():
                return self.get_leaf_index()
            else:
                child_index = int(self._test(x, params=self._test_params))
                return self._children[child_index].classify(x)
            
            
    def get_number_of_samples(self):
        return len(self._dat_ref)
            
            
    def split(self, split_params):
        """
        Applies a split.
        """
        
        self._split_params = split_params
        
        # re-calculate labels and transitions for split
        new_labels, new_dat_refs, new_trans = split_params.calc_labels_and_transitions_matrices()
        
        # copy labels and transitions to model
        action = split_params._action
        self._model._partitionings[action] = self._model._partitionings[action]._replace(labels = new_labels, transitions = new_trans) 
        
        # copy new references to children
        child_1, child_2 = super(WorldmodelTree, self).split(model=self._model)
        child_1._dat_ref = new_dat_refs[0]
        child_2._dat_ref = new_dat_refs[1]
        
        # free some memory
        self._dat_ref = None
        
        return child_1, child_2
    

    def _get_data_refs(self):
        """
        Returns the data references (i.e. indices for root.data) belonging to 
        the node. If the node isn't a leaf, the data of sub-nodes is returned.
        """

        if self.is_leaf():
            return self._dat_ref

        # else        
        data_refs = []
        for child in self._children:
            data_refs += child._get_data_refs()
        
        data_refs.sort()
        assert len(data_refs) == len(set(data_refs))
        return data_refs


    def _get_transition_refs(self, heading_in=False, inside=True, heading_out=False):
        """
        Finds all transitions that start, end or happen strictly inside the 
        node. The result is given as two lists of references. One for the start 
        and one for the end of the transition.
        """
        
        refs = self._get_data_refs()
        N = self._model.get_number_of_samples()
        
        refs_1 = [] 
         
        if heading_in:
            refs_1 += [ref-1 for ref in refs if (ref-1 not in refs) and (ref-1 > 0)]
            
        if inside:
            refs_1 += [ref for ref in refs if (ref+1 in refs)]

        if heading_out:
            refs_1 += [ref for ref in refs if (ref+1 not in refs) and (ref+1 < N)]
            
        refs_1.sort()
        refs_2 = [t+1 for t in refs_1]
        
        assert len(refs_1) == len(set(refs_1))
        return [refs_1, refs_2]
    
    
    def _reached_number_of_active_and_inactive_samples(self, number, active_action):
        """
        Calculates whether for the active _action and all other actions a certain
        number of samples is reached.
        """
        refs_1, _ = self._get_transition_refs(heading_in=False, inside=True, heading_out=False)
        actions = [self._model._actions[ref] for ref in refs_1]
        
        n_actions = len(self._model.get_known_actions())
        n_samples = len(refs_1)
        n_samples_active = actions.count(active_action)
        n_samples_inactive = n_samples - n_samples_active
        
        return (n_samples_active >= number) and (n_actions == 1 or n_samples_inactive >= number)
        
        
    def _calc_local_gain(self, active_action, test_params):
        """
        For every _action a 2x2 transition matrix is calculated, induced by the
        given split (test_params). For the "active" _action the mutual 
        information is calculated and the average of all the others. For the
        final value, mutual information of active and inactive actions each have
        half of the weight.
        
        For the transition matrices calculated, +1 is added for every possible
        transition to account for uncertainty in cases where only few samples 
        have been collected.
        """
        
        # initialize transition matrices
        matrices = {}
        actions = self._model.get_known_actions()
        for action in actions:
            matrices[action] = np.ones((2, 2))

        # transitions inside current partition
        refs_1, refs_2 = self._get_transition_refs(heading_in=False, inside=True, heading_out=False)
        refs = list(set(refs_1 + refs_2))
        refs.sort()
        
        # assign data to one of the two sub-partitions
        child_indices = [self._test(self._model._data[ref], test_params) for ref in refs]
        child_indices_1 = [child_indices[i] for i, ref in enumerate(refs) if ref in refs_1]
        child_indices_2 = [child_indices[i] for i, ref in enumerate(refs) if ref in refs_2]
        ref_test_dict = dict(zip(refs, child_indices))
        assert len(refs_1) == len(child_indices_1)
        assert len(refs_2) == len(child_indices_2)
        
        # transition matrices
        for i, ref in enumerate(refs_1):
            c1 = child_indices_1[i]
            c2 = child_indices_2[i]
            a = self._model._actions[ref]
            matrices[a][c1, c2] += 1
            
        # mutual information
        mi = entropy_utils.mutual_information(matrices[active_action])
        if len(actions) >= 2:
            mi_inactive = np.mean([entropy_utils.mutual_information(matrices[action]) for action in actions if action is not active_action])
            mi = np.mean([mi, mi_inactive])
            
        return mi, ref_test_dict
            
        

if __name__ == '__main__':
    pass
