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
        return self.get_leaves().index(self)
    
    
    def add_leaf(self):
        assert len(self.children) <= 2
        if len(self.children) >= 2:
            return None
        new_child = Tree()
        new_child.parent = self
        self.children.append(new_child)
        return new_child
    
    
    def delete(self):
        while len(self.children) > 0:
            self.children[0].delete()
        assert len(self.children) == 0
        if self.parent is not None:
            self.parent.children.remove(self)
            self.parent = None
        return


if __name__ == '__main__':
    pass