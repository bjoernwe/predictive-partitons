import numpy as np

import worldmodel_tree


class WorldmodelTrivial(WorldmodelTree):
    """
    Partitions the feature space into regular (hyper-) cubes.
    """
    
    def __init__(self, model, parents=None):
        super(WorldmodelTrivial, self).__init__(model=model)
        self.mins = None
        self.maxs = None    

    
    def _init_test(self, action, fast_partition=False):
        """
        Initializes the parameters that split the node in two halves.
        """
        assert self.status == 'leaf'
        
        # init borders
        D = self.model.get_input_dim()
        if self.mins is None:
            if len(self.parents) > 0:
                # calculate borders from parent
                parent = self.parents[0]
                self.mins = np.array(parent.mins)
                self.maxs = np.array(parent.maxs)
                dim = parent.classifier[0]
                cut = parent.classifier[1]
                # are we the first or the second child?
                assert self in parent._children
                if self is parent._children[0]:
                    self.maxs[dim] = cut
                else:
                    self.mins[dim] = cut
            else: 
                # top node
                self.mins = np.zeros(D)
                self.maxs = np.ones(D) 

        # classifier
        diffs = self.maxs - self.mins
        dim = np.argmax(diffs)
        cut = self.mins[dim] + (self.maxs[dim] - self.mins[dim]) / 2.
        #self.classifier = (dim, cut)
        return (dim, cut)


    def _test(self, x):
        """
        Tests to which child the data point x belongs
        """
        dim = self.classifier[0]
        cut = self.classifier[1]
        if x[dim] > cut:
            return 1
        return 0



if __name__ == '__main__':
    pass