import collections
import numpy as np
import scipy.linalg

import mdp

import worldmodel_tree


class WorldmodelTrivial(worldmodel_tree.WorldmodelTree):
    """
    Partitions the feature space into regular (hyper-) cubes.
    """
    
    
    TestParams = collections.namedtuple('TestParams', ['dim', 'cut'])
    
    
    def __init__(self, partitioning):
        super(WorldmodelTrivial, self).__init__(partitioning=partitioning)
        self._minima = None
        self._maxima = None    

    
    def _calc_test_params(self, active_action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves.
        """

        # init borders
        D = self.model.get_input_dim()
        if self._minima is None:
            if self._parent is not None:
                # calculate borders from parent
                parent = self._parent
                self._minima = np.array(parent._minima)
                self._maxima = np.array(parent._maxima)
                dim = parent._split_params._test_params[0]
                cut = parent._split_params._test_params[1]
                # are we the first or the second child?
                assert self in parent._children
                if self is parent._children[0]:
                    self._maxima[dim] = cut
                else:
                    self._minima[dim] = cut
            else: 
                # top node
                self._minima = np.zeros(D)
                self._maxima = np.ones(D) 

        # classifier
        diffs = self._maxima - self._minima
        dim = np.argmax(diffs)
        cut = self._minima[dim] + (self._maxima[dim] - self._minima[dim]) / 2.
        return WorldmodelTrivial.TestParams(dim=dim, cut=cut)


    def _test(self, x, params):
        """
        Tests to which child the data point x belongs.
        """
        if x[params.dim] > params.cut:
            return 1
        return 0



class WorldmodelSFA(worldmodel_tree.WorldmodelTree):
    """
    TODO: when there is only one action, search for the slowest feature
    """
    
    
    TestParams = collections.namedtuple('TestParams', ['m', 'u'])
    
    
    def __init__(self, partitioning):
        super(WorldmodelSFA, self).__init__(partitioning=partitioning)
        
        
    def _create_covariance_matrix(self, uncertainty_prior):
        cov = mdp.utils.CovarianceMatrix(bias=True)
        D = self.model.get_input_dim()
        E = np.eye(D)
        cov.update((uncertainty_prior/float(D)) * E)
        return cov

    
    def _calc_test_params(self, active_action, fast_partition=False):

        known_actions = self.model.get_known_actions()
        number_of_actions = len(known_actions)

        # get transition references (inside this node)        
        trans_refs_1 = self.get_transition_refs(heading_in=False, inside=True, heading_out=False)
        trans_refs_2 = trans_refs_1 + 1
        trans_refs = np.union1d(trans_refs_1, trans_refs_2)
        data = self.model.get_data_for_refs(refs=trans_refs)
        data_mean = np.mean(data, axis=0)
        N, D = data.shape
        
        # whitening matrix W
        cov = self._create_covariance_matrix(uncertainty_prior=self.model.uncertainty_prior)
        cov.update(data - data_mean)
        C, _, _ = cov.fix(center=False)
        E, U = scipy.linalg.eigh(C)
        W = np.dot(U, np.diag(E**(-.5))).dot(U.T)
        assert W.shape == (D, D)
        
        # whiten data
        data_whitened = np.dot(data - data_mean, W)
        #print np.dot(data_whitened.T, data_whitened)
        assert data_whitened.shape == (N, D)
        #assert np.allclose(np.dot(data_whitened.T, data_whitened), np.eye(D)) 

        # filter data references for actions
        trans_refs_active_1 = trans_refs_1[self.model.actions[trans_refs_1] == active_action]
        trans_refs_active_2 = trans_refs_active_1 + 1
        data_active_1 = self.model.get_data_for_refs(refs=trans_refs_active_1)
        data_active_2 = self.model.get_data_for_refs(refs=trans_refs_active_2)
        data_active_delta = data_active_2 - data_active_1
        
        # find fastest feature for active action
        data_active_delta_whitened = np.dot(data_active_delta, W)
        cov_active = self._create_covariance_matrix(uncertainty_prior=self.model.uncertainty_prior/float(number_of_actions))
        cov_active.update(data_active_delta_whitened)
        C, _, _ = cov_active.fix(center=True)
        E, U = scipy.linalg.eigh(C)
        final_covariance = U.dot(np.diag(E**-1).dot(U.T))
        
        # inactive covariances as well
        if number_of_actions >= 2:
            
            inactive_covariances = []
            
            for action in known_actions:
                
                if action == active_action:
                    continue
                
                # get references and data
                trans_refs_inactive_1 = trans_refs_1[self.model.actions[trans_refs_1] == action]
                trans_refs_inactive_2 = trans_refs_inactive_1 + 1
                data_inactive_1 = self.model.get_data_for_refs(refs=trans_refs_inactive_1)
                data_inactive_2 = self.model.get_data_for_refs(refs=trans_refs_inactive_2)
                data_inactive_delta = data_inactive_2 - data_inactive_1
                data_inactive_delta_whitened = np.dot(data_inactive_delta, W)
                
                # calculate covariance of deltas for inactive action
                cov_inactive = self._create_covariance_matrix(uncertainty_prior=self.model.uncertainty_prior/float(number_of_actions))
                cov_inactive.update(data_inactive_delta_whitened)
                C, _, _ = cov_inactive.fix(center=True)
                inactive_covariances.append(C)
                
            # calculate mean if inactive covariances
            inactive_covariance = np.mean(inactive_covariances)
            final_covariance = final_covariance.dot(inactive_covariance)
            
        # result (smallest eigenvector)
        E, U = scipy.linalg.eigh(final_covariance, eigvals=(0, 0))
        test_params = self.TestParams(m=data_mean, u=U[:])
        return test_params
                


    def _test(self, x, params):
        """
        Tests to which child the data point x belongs.
        """
        if (x - params.m).dot(params.u) > 0:
            return 1
        return 0



if __name__ == '__main__':
    pass
