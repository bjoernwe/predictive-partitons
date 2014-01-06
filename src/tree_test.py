import unittest

import tree


class Test(unittest.TestCase):


    def setUp(self):
        
        # set up an empty tree
        self.tree = tree.Tree()
        
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
        
        for i in range(2):

            # add new leaf
            new_leaf = self.tree.add_leaf()
            
            # tree is not the new leaf
            self.failIf(self.tree is new_leaf)
            
            # tree is not a leaf anymore
            self.failIf(self.tree.is_leaf())
            
            # new leaf is a leaf
            self.failUnless(new_leaf.is_leaf())
            
            # right number of leaves 
            self.failUnless(self.tree.get_number_of_leaves() == i+1)
            
            # root did not change
            self.failUnless(new_leaf.get_root() is self.tree)
            
        # add a third leaf leaf
        new_leaf = self.tree.add_leaf()

        # no third leaf        
        self.failUnless(new_leaf is None)

        # number if leaves still two
        self.failUnless(self.tree.get_number_of_leaves() == 2)

        # take newest leaf
        leaf2 = self.tree.get_leaves()[-1]

        # add two more leaves        
        for i in range(2):

            # add new leaf
            new_leaf = leaf2.add_leaf()
            
            # tree is not a leaf anymore
            self.failIf(leaf2.is_leaf())
            
            # new leaf is a leaf
            self.failUnless(new_leaf.is_leaf())
            
            # root did not change
            self.failUnless(new_leaf.get_root() is self.tree)
        
            # right number of leaves
            self.failUnless(leaf2.get_root().get_number_of_leaves() == i+2)
            
        # keep a list of leaves
        leaves = self.tree.get_leaves()
            
        # delete first leaf
        leaf1 = self.tree.get_leaves()[0]
        leaf1.delete()

        # one leaf less        
        self.failUnless(self.tree.get_number_of_leaves() == 2)

        # root has only one child now
        self.failUnless(self.tree.get_number_of_children() == 1)
        
        # delete first node with all children
        node1 = self.tree.children[0]
        node1.delete()

        # root is the last leaf now
        self.failUnless(self.tree.get_number_of_leaves() == 1)

        # root has no children left
        self.failUnless(self.tree.get_number_of_children() == 0)
        
        # all leaves deleted?
        for leaf in leaves:
            self.failUnless(leaf.parent is None)
        
            
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    