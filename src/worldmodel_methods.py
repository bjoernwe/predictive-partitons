import collections
import numpy as np

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



if __name__ == '__main__':
    pass