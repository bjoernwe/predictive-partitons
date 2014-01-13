import collections
import numpy as np

import worldmodel_tree


class WorldmodelTrivial(worldmodel_tree.WorldmodelTree):
    """
    Partitions the feature space into regular (hyper-) cubes.
    """
    
    
    TestParams = collections.namedtuple('TestParams', ['dim', 'cut'])
    
    
    def __init__(self, model, parents=None):
        super(WorldmodelTrivial, self).__init__(model=model)
        self._mins = None
        self._maxs = None    

    
    def _calc_test_params(self, active_action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves.
        """

        # init borders
        D = self._model.get_input_dim()
        if self._mins is None:
            if self._parent is not None:
                # calculate borders from parent
                parent = self._parent
                self._mins = np.array(parent._mins)
                self._maxs = np.array(parent._maxs)
                dim = parent.classifier[0]
                cut = parent.classifier[1]
                # are we the first or the second child?
                assert self in parent._children
                if self is parent._children[0]:
                    self._maxs[dim] = cut
                else:
                    self._mins[dim] = cut
            else: 
                # top node
                self._mins = np.zeros(D)
                self._maxs = np.ones(D) 

        # classifier
        diffs = self._maxs - self._mins
        dim = np.argmax(diffs)
        cut = self._mins[dim] + (self._maxs[dim] - self._mins[dim]) / 2.
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