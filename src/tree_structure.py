import weakref


class Tree(object):
    
    
    def __init__(self):
        self._children = []
        self._parent = None
        
        
    def is_leaf(self):
        if len(self._children) == 0:
            return True
        return False
    
    
    def get_root(self):
        if self._parent is None:
            return self
        else:
            return self._parent.get_root()
        
        
    def get_leaves(self):
        if len(self._children) == 0:
            return [self]
        else:
            leaves = []
            for child in self._children:
                leaves += child.get_leaves()
            return leaves
        
    
    def get_number_of_leaves(self):
        return len(self.get_leaves())
    
    
    def get_number_of_children(self):
        return len(self._children)
    
    
    def get_leaf_index(self):
        if not self.is_leaf():
            return None
        return self.get_leaves().index(self)
    
    
    def get_leaf(self, index):
        """
        Returns the leaf with the given index.
        """
        assert self.get_root() is self
        return self.get_leaves()[index]
    
    
    def split(self, **kwargs):
        """
        Creates two new instances of the same class and adds it as a children to 
        the current node. **kwargs are given to the constructor.
        """
        assert self.get_number_of_children() == 0
        assert self.is_leaf()
        child_1 = self.__class__(**kwargs)
        child_2 = self.__class__(**kwargs)
        child_1._parent = weakref.proxy(self)
        child_2._parent = weakref.proxy(self)
        self._children.append(child_1)
        self._children.append(child_2)
        return (child_1, child_2)
    
    

if __name__ == '__main__':
    pass