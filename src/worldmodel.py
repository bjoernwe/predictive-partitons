import numpy as np
import random

import worldmodel_methods
import worldmodel_tree


class Worldmodel(object):


    def __init__(self, method='naive', seed):
        
        # data storage
        self._data = None        # global data storage
        self._actions = None     # either None or a list of actions
        
        self._transitions = {}
        #self.transitions[None] = np.array([[0]], dtype=np.int)

        # root node of tree
        assert method in ['naive']
        self._method = method
        if method == 'naive':
            self._tree = WorldmodelTrivial(model=self)
        
        # random generator
        self._random = random.Random()
        if seed is not None:
            self.random.seed(seed)
    
    

if __name__ == '__main__':
    pass