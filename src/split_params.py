import numpy as np

import entropy_utils


class SplitParams(object):
    
    def __init__(self, node, action, test_params):
        self._node = node
        self._action = action
        self._test_params = test_params
        self._gain, self._ref_test_dict = self._calc_local_gain()
        self._new_labels, self._new_dat_refs = None, None
        self._new_trans = None 
        return


    def apply(self):
        self._node.split(split_params=self)
        return
    
    
    def get_new_labels(self):
        if self._new_labels is None:
            self._new_labels, self._new_dat_refs = self._relabel_data(ref_test_dict=self._ref_test_dict)
        return self._new_labels
    
    
    def get_new_dat_refs(self):
        if self._new_dat_refs is None:
            self._new_labels, self._new_dat_refs = self._relabel_data(ref_test_dict=self._ref_test_dict)
        return self._new_dat_refs
    
    
    def get_new_trans(self):
        if self._new_trans is None:
            self._new_trans = self._calc_transition_matrices()
        return self._new_trans
    
    
    def _relabel_data(self, ref_test_dict=None):
        """
        Returns new labels and split data references.
        """

        # some useful variables
        node = self._node
        partitioning = node._model._partitionings[self._action]
        current_state = node.get_leaf_index()
        assert node.is_leaf()
        assert current_state is not None
        
        # make of copy of all labels
        # increase labels above current state by one to make space for the split
        new_labels = [(label+1 if label > current_state else label) for label in partitioning.labels]
        new_dat_ref = [[], []]

        # every entry belonging to this node has to be re-classified
        for ref in node._dat_ref:
            if ref_test_dict is not None and ref in ref_test_dict:
                child_i = ref_test_dict[ref]
            else:
                dat = node._model._data[ref]
                child_i = node._test(dat, params=self._test_params)
            new_labels[ref] += child_i
            new_dat_ref[child_i].append(ref)

        assert len(new_labels) == len(partitioning.labels)
        assert len(node._dat_ref) == len(new_dat_ref[0]) + len(new_dat_ref[1])
        
        # does the split really split the data in two?
        assert len(new_dat_ref[0]) > 0
        assert len(new_dat_ref[1]) > 0
        #if (len(new_dat_ref[0]) == 0 or
        #    len(new_dat_ref[1]) == 0):
        #    return None, None
        return new_labels, new_dat_ref


    def _calc_transition_matrices(self):
        """
        Calculates a new transition matrix with the split index -> index & index+1.
        """
        
        # helper variables
        new_labels = self._new_labels
        N = len(new_labels)
        node = self._node
        index = node.get_leaf_index()
        action = self._action
        partitioning = node._model._partitionings[action]
        assert node.is_leaf()
        
        # result
        transition_matrices = {}
        
        for a in node._model.get_known_actions():
        
            # new transition matrix
            new_trans = np.array(partitioning.transitions[a])
            # split current row and set to zero
            new_trans[index,:] = 0
            new_trans = np.insert(new_trans, index, 0, axis=0)  # new row
            # split current column and set to zero
            new_trans[:,index] = 0
            new_trans = np.insert(new_trans, index, 0, axis=1)  # new column
            
            # update all transitions from or to current state
            for i in range(N-1):
                if node._model._actions[i] == a:
                    source = partitioning.labels[i]
                    target = partitioning.labels[i+1]
                    if source == index or target == index:
                        new_source = new_labels[i]
                        new_target = new_labels[i+1]
                        new_trans[new_source, new_target] += 1
    
            assert np.sum(new_trans) == np.sum(partitioning.transitions[a])
            transition_matrices[a] = new_trans
            
        return transition_matrices


    def _calc_local_gain(self):
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
        actions = self._node._model.get_known_actions()
        for action in actions:
            matrices[action] = np.ones((2, 2))

        # transitions inside current partition
        refs_1, refs_2 = self._node._get_transition_refs(heading_in=False, inside=True, heading_out=False)
        refs = list(set(refs_1 + refs_2))
        refs.sort()
        
        # assign data to one of the two sub-partitions
        child_indices = [self._node._test(self._node._model._data[ref], self._test_params) for ref in refs]
        child_indices_1 = [child_indices[i] for i, ref in enumerate(refs) if ref in refs_1]
        child_indices_2 = [child_indices[i] for i, ref in enumerate(refs) if ref in refs_2]
        ref_test_dict = dict(zip(refs, child_indices))
        assert len(refs_1) == len(child_indices_1)
        assert len(refs_2) == len(child_indices_2)
        
        # transition matrices
        for i, ref in enumerate(refs_1):
            c1 = child_indices_1[i]
            c2 = child_indices_2[i]
            a = self._node._model._actions[ref]
            matrices[a][c1, c2] += 1
            
        # mutual information
        mi = entropy_utils.mutual_information(matrices[self._action])
        if len(actions) >= 2:
            mi_inactive = np.mean([entropy_utils.mutual_information(matrices[action]) for action in actions if action is not self._action])
            mi = np.mean([mi, mi_inactive])
            
        return mi, ref_test_dict
    
    
    def _calc_global_gain(self):
        pass



if __name__ == '__main__':
    pass