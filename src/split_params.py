import numpy as np
import weakref

import entropy_utils


class SplitParamsLocalGain(object):
    
    def __init__(self, node):
        self._node = self._get_weakref_proxy(node)
        self._model = self._get_weakref_proxy(node.model)
        self._partitioning = self._get_weakref_proxy(node._partitioning)
        self._active_action = node._active_action
        self._test_params = node._calc_test_params(active_action=self._active_action)
        self._gain = None
        self._new_labels = None
        self._new_data_refs = None
        self._new_trans = None
        self._transition_children = None
        self._transition_refs = None
        self._transition_refs_1 = None
        self._non_transition_children = None
        self._non_transition_refs = None
        self._number_of_samples_when_updated = 0 
        return
    
    
    def _get_weakref_proxy(self, ref):
        if type(ref) in weakref.ProxyTypes:
            return ref
        return weakref.proxy(ref)
    
    
    def apply(self):
        self._node.split(split_params=self)
        return
    

    #@profile    
    def _init_transition_children(self):
        """
        Initializes the list of child indices belonging to transitions inside
        the node, i.e., a list containing 0 or 1 for every reference.
        """
        assert self._transition_refs is None
        assert self._transition_refs_1 is None
        assert self._transition_children is None
        
        # some useful variables
        data = self._node.model.data
        test_function = self._node._test
        test_params = self._test_params
        assert self._node.is_leaf()

        # create a vectorized test-function
        t = lambda r: test_function(data[r], params=test_params)
        t = np.vectorize(t, otypes=[np.int])

        # every entry of that node has to be classified...
        refs_1 = self._node.get_transition_refs(heading_in=False, inside=True, heading_out=False)
        refs_2 = refs_1 + 1
        refs = np.union1d(refs_1, refs_2)
        
        if len(refs) > 0:
            self._transition_children = t(refs)
        else:
            self._transition_children = np.empty(0, dtype=int)
        
        # store references for later
        self._transition_refs = refs
        self._transition_refs_1 = refs_1
        self._number_of_samples_when_updated = self._node.get_number_of_samples()
        return
        
        
    def _init_non_transition_children(self):
        """
        Initializes the list of child indices not belonging to transitions 
        inside the node, i.e., a list containing 0 or 1 for every reference.
        """
        assert self._non_transition_refs is None
        assert self._non_transition_children is None
        
        # some useful variables
        data = self._node.model.data
        test_function = self._node._test
        test_params = self._test_params
        assert self._node.is_leaf()

        # create a vectorized test-function
        t = lambda r: test_function(data[r], params=test_params)
        t = np.vectorize(t, otypes=[np.int])

        # calculate indices for non-transition references
        all_refs = self._node.get_data_refs()
        transition_refs = self._transition_refs
        if len(transition_refs) < len(all_refs):
            self._non_transition_refs = np.setdiff1d(all_refs, transition_refs, assume_unique=False)
            self._non_transition_children = t(self._non_transition_refs)
        else:
            self._non_transition_refs = np.empty(0, dtype=int)
            self._non_transition_children = np.empty(0, dtype=int)
        
        assert self._node.get_number_of_samples() == len(self._transition_refs) + len(self._non_transition_refs)
        return
        
        
    def update(self):
        """
        In case of new transitions inside the node, those are classified
        """
        
        # new samples?
        if self._number_of_samples_when_updated == self._node.get_number_of_samples():
            return
            
        # new transitions?
        current_transitions_refs_1 = self._node.get_transition_refs(heading_in=False, inside=True, heading_out=False)
        if self._transition_refs_1 is None or len(current_transitions_refs_1) == len(self._transition_refs_1):
            return
        
        # new transitions! also update test parameters
        self._test_params = self._node._calc_test_params(active_action=self._active_action)
        
        # calculate new transitions
        current_transitions_refs_2 = current_transitions_refs_1 + 1
        current_transitions_refs = np.union1d(current_transitions_refs_1, current_transitions_refs_2)
        new_transition_refs = np.setdiff1d(current_transitions_refs, self._transition_refs, assume_unique=True)

        # create a vectorized test-function
        data = self._model.data
        test_function = self._node._test
        test_params = self._test_params
        t = lambda r: test_function(data[r], params=test_params)
        t = np.vectorize(t, otypes=[np.int])
        
        # classify new references
        new_transition_children = t(new_transition_refs)
        
        # store results
        self._transition_refs = current_transitions_refs
        self._transition_refs_1 = current_transitions_refs_1
        self._transition_children = np.hstack([self._transition_children, new_transition_children])
        assert len(self._transition_refs) == len(self._transition_children)
        
        # reset
        self._gain = None
        self._number_of_samples_when_updated = self._node.get_number_of_samples()
        return False
    
    


    #@profile
    def get_gain(self):
        """
        For every model_action a 2x2 transition matrix is calculated, induced by 
        the given split (test_params). For the "active" model_action the mutual 
        information is calculated and the average of all the others. For the
        final value, mutual information of active and inactive actions each have
        half of the weight.
         
        For the transition matrices calculated, +10 is added for every possible
        transition to account for uncertainty in cases where only few samples 
        have been collected.
        """
        
        if self._gain is not None:
            return self._gain
        
        if self._test_params is None:
            return 0.0
        
        self._init_transition_children()
         
        # helper variables
        known_actions = self._model.get_known_actions()
        refs = self._transition_refs
        refs_1 = self._transition_refs_1
        refs_2 = refs_1 + 1
        indices_1 = self._transition_children[np.in1d(refs, refs_1, assume_unique=True)]
        indices_2 = self._transition_children[np.in1d(refs, refs_2, assume_unique=True)]
         
        # transition matrices
        matrices = {}
        for action in known_actions:
            action_mask = (self._model.actions[refs_1] == action)
            matrices[action] = np.ones((2, 2)) * self._model.uncertainty_prior
            matrices[action][0,0] += np.count_nonzero((indices_1 == 0) & (indices_2 == 0) & action_mask)
            matrices[action][0,1] += np.count_nonzero((indices_1 == 0) & (indices_2 == 1) & action_mask)
            matrices[action][1,0] += np.count_nonzero((indices_1 == 1) & (indices_2 == 0) & action_mask)
            matrices[action][1,1] += np.count_nonzero((indices_1 == 1) & (indices_2 == 1) & action_mask)
             
        # mutual information
        mi = entropy_utils.mutual_information(matrices[self._active_action])
        if len(known_actions) >= 2:
            mi_inactive = np.mean([entropy_utils.mutual_information(matrices[action]) for action in known_actions if action is not self._active_action])
            mi = np.mean([mi, mi_inactive])
           
        self._gain = mi  
        return mi


    def get_new_labels(self):
        """
        Calculates new labels.
        """
        if self._new_labels is not None:
            return self._new_labels

        self._init_non_transition_children()
        current_state = self._node.get_leaf_index()
        new_labels = np.array(self._partitioning.labels, dtype=int)
        new_labels = np.where(new_labels > current_state, new_labels + 1, new_labels)
        new_labels[self._transition_refs] += self._transition_children
        new_labels[self._non_transition_refs] += self._non_transition_children
        
        self._new_labels = new_labels
        return new_labels
    
    
    def get_new_data_refs(self):
        """
        Calculates new data references and stores two lists, one for each child.
        """
        
        if self._new_data_refs is not None:
            return self._new_data_refs

        transition_refs = self._transition_refs
        non_transition_refs = self._non_transition_refs

        transition_refs_0 = transition_refs[self._transition_children == 0]
        transition_refs_1 = transition_refs[self._transition_children == 1]
        
        non_transition_refs_0 = non_transition_refs[self._non_transition_children == 0]
        non_transition_refs_1 = non_transition_refs[self._non_transition_children == 1]
        
        # result
        result_refs = [None, None]
        result_refs[0] = np.union1d(transition_refs_0, non_transition_refs_0)
        result_refs[1] = np.union1d(transition_refs_1, non_transition_refs_1)
        assert self._node.get_number_of_samples() == len(result_refs[0]) + len(result_refs[1])        
         
        # does the split really split the data into two parts?
        #assert len(result_refs[0]) > 0
        #assert len(result_refs[1]) > 0
                 
        self._new_data_refs = result_refs
        return result_refs


    def get_new_transition_matrices(self):
        """
        Calculates action new transition matrix with the split index -> index & index+1.
        """
        
        if self._new_trans is not None:
            return self._new_trans
         
        # helper variables
        new_labels = self.get_new_labels()
        refs = self._node.get_data_refs()
        index_1 = self._node.get_leaf_index()
        index_2 = index_1 + 1
        number_of_samples = self._model.get_number_of_samples()
        assert self._node.is_leaf()
 
        # result
        transition_matrices = {}
 
        for action in self._model.get_known_actions():
          
            # new transition matrix
            new_trans = np.array(self._partitioning.transitions[action])
            # split current row and set to zero
            new_trans[index_1,:] = 0
            new_trans = np.insert(new_trans, index_1, 0, axis=0)  # new row
            # split current column and set to zero
            new_trans[:,index_1] = 0
            new_trans = np.insert(new_trans, index_1, 0, axis=1)  # new column
 
            # transitions from current state to another
             
            refs_1 = np.setdiff1d(refs, [number_of_samples-1], assume_unique=True) # remove N-1
            refs_2 = refs_1 + 1
         
            labels_1 = new_labels[refs_1]
            labels_2 = new_labels[refs_2]
 
            mask_actions = self._model.actions[refs_1] == action
             
            for i in range(self._node.get_root().get_number_of_leaves()+1):
                new_trans[index_1, i] = np.count_nonzero((labels_1 == index_1) & (labels_2 == i) & mask_actions) 
                new_trans[index_2, i] = np.count_nonzero((labels_1 == index_2) & (labels_2 == i) & mask_actions) 
         
            # transitions into current state
             
            refs_2 = np.setdiff1d(refs, [0], assume_unique=True)
            refs_1 = refs_2 - 1
     
            labels_1 = new_labels[refs_1]
            labels_2 = new_labels[refs_2]
 
            mask_actions = self._model.actions[refs_1] == action
          
            for i in range(self._node.get_root().get_number_of_leaves()+1):
                new_trans[i, index_1] = np.count_nonzero((labels_1 == i) & (labels_2 == index_1) & mask_actions) 
                new_trans[i, index_2] = np.count_nonzero((labels_1 == i) & (labels_2 == index_2) & mask_actions) 
         
            assert np.sum(new_trans) == np.sum(self._partitioning.transitions[action])
            transition_matrices[action] = new_trans
             
        self._new_trans = transition_matrices
        return transition_matrices



if __name__ == '__main__':
    pass
