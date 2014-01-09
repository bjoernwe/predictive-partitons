import numpy as np

import tree_structure


class WorldmodelTree(tree_structure.Tree):
    
    def __init__(self, model):
        super(WorldmodelTree, self).__init__()
        
        # important references
        self._model = model
        self._dat_ref = []   # indices of data belonging to this node
        
        
    def _init_test_params(self, action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves and returns
        them.
        """
        raise NotImplementedError("Use subclass like WorldmodelSpectral instead.")


    def _test(self, x, params):
        """
        Tests to which child the data point x belongs. Parameters are the ones
        calculated by _init_test_params().
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
            
            
    def split(self):
        child_1, child_2 = super(WorldmodelTree, self).split(model=self._model)
        return child_1, child_2
    
    
        

if __name__ == '__main__':
    pass