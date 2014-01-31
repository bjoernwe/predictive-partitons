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
        self._transition_indices = None
        self._transition_refs = None
        self._transition_refs_1 = None
        self._non_transition_indices = None
        self._non_transition_refs = None
        #self._number_of_samples_when_created = len(self._node.get_data_refs()) 
        return
    
    
    def _get_weakref_proxy(self, ref):
        if type(ref) in weakref.ProxyTypes:
            return ref
        return weakref.proxy(ref)
    
    
    def apply(self):
        self._node.split(split_params=self)
        return
    
    
#     def _has_new_samples(self):
#         """
#         True when relevant data for the node has changed and updates are due.
#         """
#         return self._number_of_samples_when_created < len(self._node.get_data_refs())
    
    
    def _init_transition_indices(self):
        """
        Initializes the list of child indices belonging to transitions inside
        the node, i.e., a list containing 0 or 1 for every reference.
        """
        assert self._transition_refs is None
        assert self._transition_indices is None
        
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
        self._transition_indices = t(refs)
        
        # store references for later
        self._transition_refs = refs
        self._transition_refs_1 = refs_1
        return
        
        
    def _init_non_transition_indices(self):
        """
        Initializes the list of child indices not belonging to transitions 
        inside the node, i.e., a list containing 0 or 1 for every reference.
        """
        assert self._non_transition_refs is None
        assert self._non_transition_indices is None
        
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
            self._non_transition_indices = t(self._non_transition_refs)
        else:
            self._non_transition_refs = np.empty(0, dtype=int)
            self._non_transition_indices = np.empty(0, dtype=int)
        
        assert self._node.get_number_of_samples() == len(self._transition_refs) + len(self._non_transition_refs)
        return
        
        
#     def _update_child_indices(self):
#         """
#         Updates the list of child indices, i.e., the list containing 0 or 1 
#         for every reference belonging to a transition inside the node.
#         """
#         assert self._transition_indices is not None
#         assert self._has_new_samples()
#         
#         # some useful variables
#         data = self._node.model.data
#         current_state = self._node.get_leaf_index()
#         test_function = self._node._test
#         test_params = self._test_params
#         assert self._node.is_leaf()
#         assert current_state is not None
#         
#         N = len(self.)
# 
#         # create a vectorized test-function
#         t = lambda r: test_function(data[r], params=test_params)
#         t = np.vectorize(t, otypes=[np.int])
# 
#         # every new entry of that node has to be classified...
#         refs_1 = self._node.get_data_refs()[M:]
#         refs_2 = refs_1 + 1
#         refs = np.union1d(refs_1, refs_2) 
#         new_child_indices = t(refs)
#         self._transition_indices = np.hstack([self._transition_indices, new_child_indices])
#             
#         return


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
        
        self._init_transition_indices()
         
        # helper variables
        known_actions = self._model.get_known_actions()
        refs = self._transition_refs
        refs_1 = self._transition_refs_1
        refs_2 = refs_1 + 1
        indices_1 = self._transition_indices[np.in1d(refs, refs_1, assume_unique=True)]
        indices_2 = self._transition_indices[np.in1d(refs, refs_2, assume_unique=True)]
         
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

        self._init_non_transition_indices()
        current_state = self._node.get_leaf_index()
        new_labels = np.array(self._partitioning.labels, dtype=int)
        new_labels = np.where(new_labels > current_state, new_labels + 1, new_labels)
        new_labels[self._transition_refs] += self._transition_indices
        new_labels[self._non_transition_refs] += self._non_transition_indices
        
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

        transition_refs_0 = transition_refs[self._transition_indices == 0]
        transition_refs_1 = transition_refs[self._transition_indices == 1]
        
        non_transition_refs_0 = non_transition_refs[self._non_transition_indices == 0]
        non_transition_refs_1 = non_transition_refs[self._non_transition_indices == 1]
        
        # result
        result_refs = [None, None]
        result_refs[0] = np.union1d(transition_refs_0, non_transition_refs_0)
        result_refs[1] = np.union1d(transition_refs_1, non_transition_refs_1)
        assert self._node.get_number_of_samples() == len(result_refs[0]) + len(result_refs[1])        
         
        # does the split really split the data into two parts?
        assert len(result_refs[0]) > 0
        assert len(result_refs[1]) > 0
                 
        self._new_data_refs = result_refs
        return result_refs


#     def get_gain(self):
#         if self._gain is None:
#             self._gain = self._calc_local_gain()
#         return self._gain
    
    
#     def get_new_labels(self):
#         if self._new_labels is None or True in (self._new_labels < 0):
#             self._update_labels()
#         return self._new_labels
    
    
#     def get_new_dat_refs(self):
#         if self._new_dat_refs is None:
#             self._update_dat_refs()
#         return self._new_dat_refs
    
    
#     def get_new_trans(self):
#         if self._new_trans is None:
#             self._update_transition_matrices()
#         return self._new_trans
    

#     def _init_new_labels(self):    
#         
#         current_state = self._node.get_leaf_index()
#         assert self._node.is_leaf()
#         assert current_state is not None
#         
#         if self._new_labels is None:
#             new_labels = np.array(self._partitioning.labels, dtype=int)
#             new_labels = np.where(new_labels > current_state, new_labels + 1, new_labels)
#             new_labels[new_labels == current_state] = -1
#             self._new_labels = new_labels
#             
#         return


#     def _update_labels(self):
#         """
#         Calculates new labels.
#         """
# 
#         # some useful variables
#         data = self._node.model.data
#         current_state = self._node.get_leaf_index()
#         test_function = self._node._test
#         test_params = self._test_params
#         assert self._node.is_leaf()
#         assert current_state is not None
# 
#         # create a vectorized test-function
#         t = lambda r: test_function(data[r], params=test_params)
#         t = np.vectorize(t, otypes=[np.int])
# 
#         self._init_new_labels()        
#         new_labels = self._new_labels
#         
#         # TODO: We can avoid re-calculations of (old) labels, when test-function
#         # has not changed. We could consider caching the test-parameters...
#         
#         # every entry that node has to be re-classified...
#         refs = np.where(new_labels==-1)[0]
#         child_indices = t(refs)
#         new_labels[refs] = current_state + child_indices
#         self._new_labels = new_labels
#         
#         assert np.count_nonzero(self._new_labels==-1) == 0
#         assert len(self._new_labels) == len(self._partitioning.labels)
#         return
    
    
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



# class SplitParams(object):
#     
#     def __init__(self, node, action, test_params):
#         self.model_action = action
#         self._node = weakref.proxy(node)
#         self._model = self._get_weakref_proxy(self._node.model)
#         self._partitioning = self._get_weakref_proxy(self._model.partitionings[self.model_action])
#         self._test_params = test_params
#         self._gain = None
#         self._new_labels = None
#         self._new_dat_refs = None
#         self._new_trans = None
#         self._number_of_samples_when_created = len(self._node.get_data_refs()) 
#         return
#     
#     
#     def _get_weakref_proxy(self, ref):
#         if type(ref) in weakref.ProxyTypes:
#             return ref
#         return weakref.proxy(ref)
#     
#     
#     def apply(self):
#         self._node.split(split_params=self)
#         return
#     
#     
#     def get_gain(self):
#         if self._gain is None:
#             if self._node.model.gain_measure == 'local':
#                 self._gain = self._calc_local_gain()
#             elif self._node.model.gain_measure == 'global':
#                 self._gain = self._calc_global_gain()
#             else:
#                 assert False
#         return self._gain
#     
#     
#     def get_new_labels(self):
#         if self._new_labels is None or True in (self._new_labels < 0):
#             self._update_labels()
#         return self._new_labels
#     
#     
#     def get_new_dat_refs(self):
#         if self._new_dat_refs is None:
#             self._update_dat_refs()
#         return self._new_dat_refs
#     
#     
#     def get_new_trans(self):
#         if self._new_trans is None:
#             self._update_transition_matrices()
#         return self._new_trans
#     
# 
#     def _init_new_labels(self):    
#         
#         current_state = self._node.get_leaf_index()
#         assert self._node.is_leaf()
#         assert current_state is not None
#         
#         if self._new_labels is None:
#             new_labels = np.array(self._partitioning.labels, dtype=int)
#             new_labels = np.where(new_labels > current_state, new_labels + 1, new_labels)
#             new_labels[new_labels == current_state] = -1
#             self._new_labels = new_labels
#             
#         return
# 
# 
#     def _update_labels(self):
#         """
#         Calculates new labels.
#         """
# 
#         # some useful variables
#         data = self._node.model.data
#         current_state = self._node.get_leaf_index()
#         test_function = self._node._test
#         test_params = self._test_params
#         assert self._node.is_leaf()
#         assert current_state is not None
# 
#         # create a vectorized test-function
#         t = lambda r: test_function(data[r], params=test_params)
#         t = np.vectorize(t, otypes=[np.int])
# 
#         self._init_new_labels()        
#         new_labels = self._new_labels
#         
#         # TODO: We can avoid re-calculations of (old) labels, when test-function
#         # has not changed. We could consider caching the test-parameters...
#         
#         # every entry that node has to be re-classified...
#         refs = np.where(new_labels==-1)[0]
#         child_indices = t(refs)
#         new_labels[refs] = current_state + child_indices
#         self._new_labels = new_labels
#         
#         assert np.count_nonzero(self._new_labels==-1) == 0
#         assert len(self._new_labels) == len(self._partitioning.labels)
#         return
#     
#     
#     def _update_dat_refs(self):
#         """
#         Calculates new data references and stores two lists, one for each child.
#         """
#         
#         current_state = self._node.get_leaf_index()
#         assert current_state is not None
#         
#         new_labels = self.get_new_labels()
#         new_dat_refs = [None, None]
#         
#         assert np.count_nonzero(new_labels < 0) == 0
#         assert len(self._node.data_refs) == np.count_nonzero(new_labels == current_state) + np.count_nonzero(new_labels == current_state+1)
#         
#         new_dat_refs[0] = np.where(new_labels == current_state)[0]
#         new_dat_refs[1] = np.where(new_labels == current_state+1)[0]
#         
#         # does the split really split the data into two parts?
#         assert len(new_dat_refs[0]) > 0
#         assert len(new_dat_refs[1]) > 0
#                 
#         self._new_dat_refs = new_dat_refs
#         return
# 
# 
#     def _update_transition_matrices(self):
#         """
#         Calculates action new transition matrix with the split index -> index & index+1.
#         """
#         
#         # helper variables
#         new_labels = self.get_new_labels()
#         refs = self._node.get_data_refs()
#         index_1 = self._node.get_leaf_index()
#         index_2 = index_1 + 1
#         number_of_samples = self._model.get_number_of_samples()
#         assert self._node.is_leaf()
#         assert len(refs) == self._number_of_samples_when_created
# 
#         # result
#         transition_matrices = {}
# 
#         for action in self._model.get_known_actions():
#          
#             # new transition matrix
#             new_trans = np.array(self._partitioning.transitions[action])
#             # split current row and set to zero
#             new_trans[index_1,:] = 0
#             new_trans = np.insert(new_trans, index_1, 0, axis=0)  # new row
#             # split current column and set to zero
#             new_trans[:,index_1] = 0
#             new_trans = np.insert(new_trans, index_1, 0, axis=1)  # new column
# 
#             # transitions from current state to another
#             
#             refs_1 = np.setdiff1d(refs, [number_of_samples-1], assume_unique=True) # remove N-1
#             refs_2 = refs_1 + 1
#         
#             labels_1 = new_labels[refs_1]
#             labels_2 = new_labels[refs_2]
# 
#             mask_actions = self._model.actions[refs_1] == action
#             
#             for i in range(self._node.get_root().get_number_of_leaves()+1):
#                 new_trans[index_1, i] = np.count_nonzero((labels_1 == index_1) & (labels_2 == i) & mask_actions) 
#                 new_trans[index_2, i] = np.count_nonzero((labels_1 == index_2) & (labels_2 == i) & mask_actions) 
#         
#             # transitions into current state
#             
#             refs_2 = np.setdiff1d(refs, [0], assume_unique=True)
#             refs_1 = refs_2 - 1
#     
#             labels_1 = new_labels[refs_1]
#             labels_2 = new_labels[refs_2]
# 
#             mask_actions = self._model.actions[refs_1] == action
#          
#             for i in range(self._node.get_root().get_number_of_leaves()+1):
#                 new_trans[i, index_1] = np.count_nonzero((labels_1 == i) & (labels_2 == index_1) & mask_actions) 
#                 new_trans[i, index_2] = np.count_nonzero((labels_1 == i) & (labels_2 == index_2) & mask_actions) 
#         
#             assert np.sum(new_trans) == np.sum(self._partitioning.transitions[action])
#             transition_matrices[action] = new_trans
#             
#         self._new_trans = transition_matrices
#         return
# 
# 
#     def _calc_local_gain(self):
#         """
#         For every model_action a 2x2 transition matrix is calculated, induced by the
#         given split (test_params). For the "active" model_action the mutual 
#         information is calculated and the average of all the others. For the
#         final value, mutual information of active and inactive actions each have
#         half of the weight.
#         
#         For the transition matrices calculated, +10 is added for every possible
#         transition to account for uncertainty in cases where only few samples 
#         have been collected.
#         """
#         
#         # helper variables
#         current_state = self._node.get_leaf_index()
#         data = self._model.data
#         test_function = self._node._test
#         test_params = self._test_params
#         
#         # transitions inside current partition
#         refs_1, refs_2 = self._node.get_transition_refs(heading_in=False, inside=True, heading_out=False)
#         refs = np.union1d(refs_1, refs_2)
#         
#         # assign data to one of the two sub-partitions
#         t = lambda r: test_function(data[r], params=test_params)
#         t = np.vectorize(t, otypes=[np.int])
#         child_indices = t(refs)
#         
#         # new labels
#         self._init_new_labels()        
#         self._new_labels[refs] = current_state + child_indices
#         new_labels_1 = self._new_labels[refs_1]
#         new_labels_2 = self._new_labels[refs_1+1]
#         
#         # initialize transition matrices
#         matrices = {}
#         known_actions = self._model.get_known_actions()
#         for action in known_actions:
#             matrices[action] = np.ones((2, 2)) * self._model.uncertainty_prior
#             
#         # transition matrices
#         for action in known_actions:
#             action_mask = (self._model.actions[refs_1] == action)
#             matrices[action][0,0] += np.count_nonzero((new_labels_1 == current_state) & (new_labels_2 == current_state) & action_mask)
#             matrices[action][0,1] += np.count_nonzero((new_labels_1 == current_state) & (new_labels_2 == current_state+1) & action_mask)
#             matrices[action][1,0] += np.count_nonzero((new_labels_1 == current_state+1) & (new_labels_2 == current_state) & action_mask)
#             matrices[action][1,1] += np.count_nonzero((new_labels_1 == current_state+1) & (new_labels_2 == current_state+1) & action_mask)
#             
#         # mutual information
#         mi = entropy_utils.mutual_information(matrices[self.model_action])
#         if len(known_actions) >= 2:
#             mi_inactive = np.mean([entropy_utils.mutual_information(matrices[action]) for action in known_actions if action is not self.model_action])
#             mi = np.mean([mi, mi_inactive])
#             
#         return mi
#     
#     
#     def _calc_global_gain(self):
#         
#         active_action = self.model_action
#         actions = self._node.model.get_known_actions()
#         N = self._node.model.partitionings[active_action].tree.get_number_of_leaves()
#         
#         old_trans_uncertain = {}
#         new_trans_uncertain = {}
#         new_trans = self.get_new_trans()
#         uncertain_trans_old = np.ones((N, N), dtype=int) * self._node.model.uncertainty_prior
#         uncertain_trans_new = np.ones((N+1, N+1), dtype=int) * self._node.model.uncertainty_prior
#         
#         for action in actions:
#             old_trans_uncertain[action] = self._node.model.partitionings[active_action].transitions[action] + uncertain_trans_old
#             new_trans_uncertain[action] = new_trans[action] + uncertain_trans_new
#         
#         mi_old = entropy_utils.mutual_information(P=old_trans_uncertain[active_action])
#         mi_new = entropy_utils.mutual_information(P=new_trans_uncertain[active_action])
#         
#         if len(actions) >= 2:
#             mi_old_inactive = np.mean([entropy_utils.mutual_information(old_trans_uncertain[action]) for action in actions if action is not active_action])
#             mi_new_inactive = np.mean([entropy_utils.mutual_information(new_trans_uncertain[action]) for action in actions if action is not active_action])
#             mi_old = np.mean([mi_old, mi_old_inactive])
#             mi_new = np.mean([mi_new, mi_new_inactive])
#             
#         return mi_new - mi_old


if __name__ == '__main__':
    pass
