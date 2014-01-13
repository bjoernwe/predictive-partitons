import unittest

import tree_structure


class Test(unittest.TestCase):


    def setUp(self):
        pass
    
    
    def testTree(self):
        # set up empty tree
        tree = tree_structure.Tree()

        # root is a leaf
        self.failUnless(tree.is_leaf())
        
        # root of tree is tree itself again
        self.failUnless(tree.get_root() == tree)
        
        # first leaf is tree itself again
        self.failUnless(tree.get_leaves()[0] == tree)
        
        # number of leaves equals 1
        self.failUnless(tree.get_number_of_leaves() == 1)
        
        # index is the first leaf
        self.failUnless(tree.get_leaf_index() == 0)
        
        # add new leaf
        new_leaf_1, new_leaf_2 = tree.split()
        self.failIf(new_leaf_1 is None)
        self.failIf(new_leaf_2 is None)
        
        # tree is not the new leaf
        self.failIf(tree is new_leaf_1)
        self.failIf(tree is new_leaf_2)
        
        # tree is not a leaf anymore
        self.failIf(tree.is_leaf())
        
        # new leaf is a leaf
        self.failUnless(new_leaf_1.is_leaf())
        self.failUnless(new_leaf_2.is_leaf())
        
        # right number of leaves 
        self.failUnless(tree.get_number_of_leaves() == 2)
        
        # root did not change
        self.failUnless(new_leaf_1.get_root() is tree)
        self.failUnless(new_leaf_2.get_root() is tree)
            
        # add a new leafs
        new_leaf_3, new_leaf_4 = new_leaf_2.split()
        self.failIf(new_leaf_3 is None)
        self.failIf(new_leaf_4 is None)

        # tree is not the new leaf
        self.failIf(tree is new_leaf_3)
        self.failIf(tree is new_leaf_4)
        
        # every leaf in place?
        self.failIf(tree.is_leaf())
        self.failUnless(new_leaf_1.is_leaf())
        self.failIf(new_leaf_2.is_leaf())
        self.failUnless(new_leaf_3.is_leaf())
        self.failUnless(new_leaf_4.is_leaf())
        
        # right number of leaves 
        self.failUnless(tree.get_number_of_leaves() == 3)
        
        # root did not change
        self.failUnless(new_leaf_1.get_root() is tree)
        self.failUnless(new_leaf_2.get_root() is tree)
        self.failUnless(new_leaf_3.get_root() is tree)
        self.failUnless(new_leaf_4.get_root() is tree)

        return
        
            
        
if __name__ == "__main__":
    unittest.main()
    