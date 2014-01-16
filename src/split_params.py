import numpy as np

import entropy_utils


class SplitParams(object):
    
    def __init__(self, node, action, test_params):
        self._node = node
        self._action = action
        self._test_params = test_params
        self._gain = None
        self._new_labels = None
        self._new_dat_refs = None
        self._new_trans = None 
        return


    def apply(self):
        self._node.split(split_params=self)
        return
    
    
    def get_gain(self):
        if self._gain is None:
            if self._node.model.gain_measure == 'local':
                self._gain = self._calc_local_gain()
            elif self._node.model.gain_measure == 'global':
                self._gain = self._calc_global_gain()
            else:
                assert False
        return self._gain
    
    
    def get_new_labels(self):
        if self._new_labels is None or True in (self._new_labels < 0):
            self._update_labels()
        return self._new_labels
    
    
    def get_new_dat_refs(self):
        if self._new_dat_refs is None:
            self._update_dat_refs()
        return self._new_dat_refs
    
    
    def get_new_trans(self):
        if self._new_trans is None:
            self._update_transition_matrices()
        return self._new_trans
    

    def _init_new_labels(self):    
        
        node = self._node
        partitioning = node.model.partitionings[self._action]
        current_state = node.get_leaf_index()
        assert node.is_leaf()
        assert current_state is not None
        
        if self._new_labels is None:
            new_labels = np.array(partitioning.labels, dtype=int)
            new_labels = np.where(new_labels > current_state, new_labels + 1, new_labels)
            new_labels = np.where(new_labels == current_state, -1, new_labels)
            self._new_labels = new_labels
            
        return


    #@profile
    def _update_labels(self):
        """
        Calculates new labels. If some of them are already calculated, old
        values are kept.
        """

        # some useful variables
        node = self._node
        data = node.model.data
        partitioning = node.model.partitionings[self._action]
        current_state = node.get_leaf_index()
        test_function = node._test
        test_params = self._test_params
        assert node.is_leaf()
        assert current_state is not None

        self._init_new_labels()        
        new_labels = self._new_labels

        # every entry that node has to be re-classified...
        refs = np.where(new_labels==-1)[0]
        child_indices = np.array([test_function(data[ref], params=test_params) for ref in refs], dtype=int)
        new_labels[refs] = current_state + child_indices
        self._new_labels = new_labels
        
        assert np.count_nonzero(self._new_labels==-1) == 0
        assert len(self._new_labels) == len(partitioning.labels)
        return
    
    
    def _update_dat_refs(self):
        """
        Calculates new data references and stores two lists, one for each child.
        """
        
        node = self._node
        current_state = node.get_leaf_index()
        assert current_state is not None
        
        new_labels = self.get_new_labels()
        new_dat_refs = [set(), set()]
        
        assert np.count_nonzero(new_labels < 0) == 0
        assert len(node.data_refs) == np.count_nonzero(new_labels == current_state) + np.count_nonzero(new_labels == current_state+1)
        
        for ref in node.data_refs:
            assert new_labels[ref] in [current_state, current_state+1]
            if new_labels[ref] == current_state:
                new_dat_refs[0].add(ref)
            else:
                new_dat_refs[1].add(ref)
                
        # does the split really split the data into two parts?
        assert len(new_dat_refs[0]) > 0
        assert len(new_dat_refs[1]) > 0
                
        self._new_dat_refs = new_dat_refs
        return


    def _update_transition_matrices(self):
        """
        Calculates a new transition matrix with the split index -> index & index+1.
        """
        
        # helper variables
        new_labels = self.get_new_labels()
        node = self._node
        model = node.model
        index = node.get_leaf_index()
        number_of_samples = model.get_number_of_samples()
        partitioning = node.model.partitionings[self._action]
        assert node.is_leaf()
        
        # all potentially changed references
        changed_refs = node.get_data_refs()
        changed_refs.update([ref-1 for ref in changed_refs])
        changed_refs.difference_update([-1, number_of_samples-1])
        
        # result
        transition_matrices = {}
        
        for a in node.model.get_known_actions():
        
            # new transition matrix
            new_trans = np.array(partitioning.transitions[a])
            # split current row and set to zero
            new_trans[index,:] = 0
            new_trans = np.insert(new_trans, index, 0, axis=0)  # new row
            # split current column and set to zero
            new_trans[:,index] = 0
            new_trans = np.insert(new_trans, index, 0, axis=1)  # new column
            
            # update all transitions from or to current state
            for ref in changed_refs:
                if model.actions[ref] == a:
                    #source = partitioning.labels[ref]
                    #target = partitioning.labels[ref+1]
                    #assert source == index or target == index
                    new_source = new_labels[ref]
                    new_target = new_labels[ref+1]
                    new_trans[new_source, new_target] += 1
    
            assert np.sum(new_trans) == np.sum(partitioning.transitions[a])
            transition_matrices[a] = new_trans
            
        self._new_trans = transition_matrices
        return


    #@profile
    def _calc_local_gain(self):
        """
        For every _action a 2x2 transition matrix is calculated, induced by the
        given split (test_params). For the "active" _action the mutual 
        information is calculated and the average of all the others. For the
        final value, mutual information of active and inactive actions each have
        half of the weight.
        
        For the transition matrices calculated, +10 is added for every possible
        transition to account for uncertainty in cases where only few samples 
        have been collected.
        """
        
        # helper variables
        active_action = self._action
        node = self._node
        model = node.model
        current_state = node.get_leaf_index()
        data = model.data
        test_function = node._test
        test_params = self._test_params
        
        # transitions inside current partition
        refs_1, refs_2 = node.get_transition_refs(heading_in=False, inside=True, heading_out=False)
        refs = refs_1.union(refs_2)
        sorted_refs_1 = sorted(refs_1)
        sorted_refs = sorted(refs)
        assert type(refs_1) == set
        assert type(refs_2) == set
        
        # assign data to one of the two sub-partitions
        child_indices = [test_function(data[ref], test_params) for ref in sorted_refs]
        child_indices_1_2 = [(child_indices[i], child_indices[i+1]) for i, ref in enumerate(sorted_refs) if ref in refs_1]
        assert len(refs_1) == len(child_indices_1_2)
        
        # store _test results in labels to avoid re-calculation
        self._init_new_labels()        
        for i, ref in enumerate(sorted_refs):
            self._new_labels[ref] = current_state + child_indices[i]
        
        # initialize transition matrices
        matrices = {}
        actions = model.get_known_actions()
        for action in actions:
            matrices[action] = np.ones((2, 2)) * model.uncertainty_bias

        # transition matrices
        for i, ref in enumerate(sorted_refs_1):
            c1, c2 = child_indices_1_2[i]
            a = node.model.actions[ref]
            matrices[a][c1, c2] += 1
            
        # mutual information
        mi = entropy_utils.mutual_information(matrices[active_action])
        if len(actions) >= 2:
            mi_inactive = np.mean([entropy_utils.mutual_information(matrices[action]) for action in actions if action is not active_action])
            mi = np.mean([mi, mi_inactive])
            
        return mi
    
    
    def _calc_global_gain(self):
        
        active_action = self._action
        actions = self._node.model.get_known_actions()
        N = self._node.model.partitionings[active_action].tree.get_number_of_leaves()
        
        old_trans_uncertain = {}
        new_trans_uncertain = {}
        new_trans = self.get_new_trans()
        uncertain_trans_old = np.ones((N, N), dtype=int) * self._node.model.uncertainty_bias
        uncertain_trans_new = np.ones((N+1, N+1), dtype=int) * self._node.model.uncertainty_bias
        
        for action in actions:
            old_trans_uncertain[action] = self._node.model.partitionings[active_action].transitions[action] + uncertain_trans_old
            new_trans_uncertain[action] = new_trans[action] + uncertain_trans_new
        
        mi_old = entropy_utils.mutual_information(P=old_trans_uncertain[active_action])
        mi_new = entropy_utils.mutual_information(P=new_trans_uncertain[active_action])
        
        if len(actions) >= 2:
            mi_old_inactive = np.mean([entropy_utils.mutual_information(old_trans_uncertain[action]) for action in actions if action is not active_action])
            mi_new_inactive = np.mean([entropy_utils.mutual_information(new_trans_uncertain[action]) for action in actions if action is not active_action])
            mi_old = np.mean([mi_old, mi_old_inactive])
            mi_new = np.mean([mi_new, mi_new_inactive])
            
        return mi_new - mi_old


if __name__ == '__main__':
    pass