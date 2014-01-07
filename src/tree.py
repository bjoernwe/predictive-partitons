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
    
    
    def split(self, **kwargs):
        """
        Creates two new instances of the same class and adds it as a children to 
        the current node. **kwargs are given to the constructor.
        """
        assert self.get_number_of_children() == 0
        assert self.is_leaf()
        child_1 = self.__class__(**kwargs)
        child_2 = self.__class__(**kwargs)
        child_1._parent = self
        child_2._parent = self
        self._children.append(child_1)
        self._children.append(child_2)
        return (child_1, child_2)
    
    
#     def add_leaf(self, **kwargs):
#         """
#         Creates a new instance of the same class and adds it as a child to the
#         current node. **kwargs are given to the constructor.
#         """
#         assert len(self._children) <= 2
#         if len(self._children) >= 2:
#             return None
#         new_child = self.__class__(**kwargs)
#         new_child._parent = self
#         self._children.append(new_child)
#         return new_child
    
    
#     def delete(self):
#         while len(self._children) > 0:
#             self._children[0].delete()
#         assert len(self._children) == 0
#         if self._parent is not None:
#             self._parent._children.remove(self)
#             self._parent = None
#         return


if __name__ == '__main__':
    pass