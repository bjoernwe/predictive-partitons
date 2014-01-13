import numpy as np
import weakref


class SplitParams(object):
    
    def __init__(self, node, action, gain, test_params, ref_test_dict):
        self._node = weakref.proxy(node)
        self._action = action
        self._gain = gain
        self._test_params = test_params
        self._ref_test_dict = ref_test_dict 
        return
    
    
    def calc_labels_and_transitions_matrices(self):
        new_labels, new_dat_refs = self._relabel_data()
        new_trans = self._calc_transition_matrices(new_labels)
        return new_labels, new_dat_refs, new_trans


    def _relabel_data(self):
        """
        Returns new labels and split data references.
        """
        node = self._node
        assert node.is_leaf()

        # some useful variables
        current_state = node.get_leaf_index()
        assert current_state is not None
        
        # make of copy of all labels
        # increase labels above current state by one to make space for the split
        new_labels = [(label+1 if label > current_state else label) for label in node._model._labels]
        new_dat_ref = [[], []]

        # every entry belonging to this node has to be re-classified
        assert len(node._dat_ref) == len(self._ref_test_dict)
        for ref in node._dat_ref:
            child_i = self._ref_test_dict[ref]
            new_labels[ref] += child_i
            new_dat_ref[child_i].append(ref)

        assert len(new_labels) == len(node._model._labels)
        assert len(new_labels) == len(new_dat_ref[0]) + len(new_dat_ref[1])
        
        # does the split really split the data in two?
        assert len(new_dat_ref[0]) > 0
        assert len(new_dat_ref[1]) > 0
        #if (len(new_dat_ref[0]) == 0 or
        #    len(new_dat_ref[1]) == 0):
        #    return None, None
        return new_labels, new_dat_ref


    def _calc_transition_matrices(self, new_labels):
        """
        Calculates a new transition matrix with the split index -> index & index+1.
        """
        
        N = len(new_labels)
        action = self._action
        node = self._node
        index = node.get_index()
        assert node.is_leaf()
        
        transition_matrices = {}
        
        for a in node._model.get_known_actions():
        
            # new transition matrix
            new_trans = np.array(node._model._partitionings[action].transitions[a])
            # split current row and set to zero
            new_trans[index,:] = 0
            new_trans = np.insert(new_trans, index, 0, axis=0)  # new row
            # split current column and set to zero
            new_trans[:,index] = 0
            new_trans = np.insert(new_trans, index, 0, axis=1)  # new column
            
            # update all transitions from or to current state
            for i in range(N-1):
                source = self.labels[i]
                target = self.labels[i+1]
                if node._model._actions[i] == action:
                    if source == index or target == index:
                        new_source = new_labels[i]
                        new_target = new_labels[i+1]
                        new_trans[new_source, new_target] += 1
    
            assert np.sum(new_trans) == np.sum(node._model._partitionings[action].transitions[a])
            transition_matrices[a] = new_trans
            
        return transition_matrices



if __name__ == '__main__':
    pass