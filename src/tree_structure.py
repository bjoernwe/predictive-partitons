import weakref


class Tree(object):
    
    
    def __init__(self):
        self.children = []
        self.parent = None
        
        
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        return False
    
    
    def get_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()
        
        
    def get_leaves(self):
        if len(self.children) == 0:
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves += child.get_leaves()
            return leaves
        
    
    def get_number_of_leaves(self):
        return len(self.get_leaves())
    
    
    def get_number_of_children(self):
        return len(self.children)
    
    
    def get_leaf_index(self):
        if not self.is_leaf():
            return None
        return self.get_root().get_leaves().index(self)
    
    
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
        child_1.parent = weakref.proxy(self)
        child_2.parent = weakref.proxy(self)
        self.children.append(child_1)
        self.children.append(child_2)
        return (child_1, child_2)
    
    

if __name__ == '__main__':
    pass
