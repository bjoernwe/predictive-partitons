import unittest

import tree
import worldmodel_tree


class Test(unittest.TestCase):


    def setUp(self):
        
        # set up an empty tree
        self.tree = tree.Tree()
        #self.tree = worldmodel_tree.WorldmodelTree(model=self)
        
        # root is a leaf
        self.failUnless(self.tree.is_leaf())
        
        # root of tree is tree itself again
        self.failUnless(self.tree.get_root() == self.tree)
        
        # first leaf is tree itself again
        self.failUnless(self.tree.get_leaves()[0] == self.tree)
        
        # number of leaves equals 1
        self.failUnless(self.tree.get_number_of_leaves() == 1)
        
        # index is the first leaf
        self.failUnless(self.tree.get_leaf_index() == 0)
        
        return


    def testAddingRemoving(self):
        
        # add new leaf
        new_leaf_1, new_leaf_2 = self.tree.split()
        self.failIf(new_leaf_1 is None)
        self.failIf(new_leaf_2 is None)
        
        # tree is not the new leaf
        self.failIf(self.tree is new_leaf_1)
        self.failIf(self.tree is new_leaf_2)
        
        # tree is not a leaf anymore
        self.failIf(self.tree.is_leaf())
        
        # new leaf is a leaf
        self.failUnless(new_leaf_1.is_leaf())
        self.failUnless(new_leaf_2.is_leaf())
        
        # right number of leaves 
        self.failUnless(self.tree.get_number_of_leaves() == 2)
        
        # root did not change
        self.failUnless(new_leaf_1.get_root() is self.tree)
        self.failUnless(new_leaf_2.get_root() is self.tree)
            
        # add a new leafs
        new_leaf_3, new_leaf_4 = new_leaf_2.split()
        self.failIf(new_leaf_3 is None)
        self.failIf(new_leaf_4 is None)

        # tree is not the new leaf
        self.failIf(self.tree is new_leaf_3)
        self.failIf(self.tree is new_leaf_4)
        
        # every leaf in place?
        self.failIf(self.tree.is_leaf())
        self.failUnless(new_leaf_1.is_leaf())
        self.failIf(new_leaf_2.is_leaf())
        self.failUnless(new_leaf_3.is_leaf())
        self.failUnless(new_leaf_4.is_leaf())
        
        # right number of leaves 
        self.failUnless(self.tree.get_number_of_leaves() == 3)
        
        # root did not change
        self.failUnless(new_leaf_1.get_root() is self.tree)
        self.failUnless(new_leaf_2.get_root() is self.tree)
        self.failUnless(new_leaf_3.get_root() is self.tree)
        self.failUnless(new_leaf_4.get_root() is self.tree)

        return
        
            
        
if __name__ == "__main__":
    unittest.main()
    