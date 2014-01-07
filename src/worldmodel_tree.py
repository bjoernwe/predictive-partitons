import tree


class WorldmodelTree(tree.Tree):
    
    def __init__(self, model):
        super(WorldmodelTree, self).__init__()
        
        # important references
        self._model = model
        self._dat_ref = []   # indices of data belonging to this node
        
        
    def add_leaf(self):
        new_child = super(WorldmodelTree, self).add_leaf(model=self._model)
        return new_child
        
    

if __name__ == '__main__':
    pass