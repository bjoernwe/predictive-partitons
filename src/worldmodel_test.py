#! /usr/bin/python

import numpy as np
import unittest

import worldmodel


class DataTests(unittest.TestCase):

    def testNChain1(self):
        n = 100
        data = worldmodel.problemChain(n)
        self.assertEqual(data.shape[0], n)
        
    def testNChain2(self):
        n = 10000
        data = worldmodel.problemChain(n)
        self.assertEqual(data.shape[0], n)

    def testChainRandom(self):
        data = worldmodel.problemChain(n=1)
        x1 = list(data[0])
        data = worldmodel.problemChain(n=1)
        x2 = list(data[0])
        self.assertNotEqual(x1, x2)

    def testChainSeed(self):
        data = worldmodel.problemChain(n=1, seed=0)
        x1 = list(data[0])
        data = worldmodel.problemChain(n=1, seed=0)
        x2 = list(data[0])
        self.assertEqual(x1, x2)


class TrivialTransitionsTest(unittest.TestCase):

    def testTrivialTransMatrix(self):
        data = worldmodel.problemChain(n=1000, seed=0)
        tree = worldmodel.WorldModelTree()
        tree.add_data(data)
        trans = tree._merge_transition_matrices()
        self.assertEqual(trans.shape, (1,1))
        self.assertEqual(trans[0,0], data.shape[0]-1) # n of transitions

        

class TrivialEntropyTests(unittest.TestCase):

    def setUp(self):
        self.data = worldmodel.problemChain(n=1000, seed=0)

    def testTrivialEntropy(self):
        tree = worldmodel.WorldModelTree()
        tree.add_data(self.data)
        entropy = tree.entropy()
        self.assertEqual(entropy, 1.0)

    def testTrivialTransEntropy(self):
        tree = worldmodel.WorldModelTree()
        tree.add_data(self.data)
        transitions = tree._merge_transition_matrices()
        entropy = worldmodel.WorldModelTree._matrix_entropy(transitions=transitions)
        self.assertEqual(entropy, 1.0)

    def testTrivialLeafEntropy(self):
        tree = worldmodel.WorldModelTree()
        tree.add_data(self.data)
        entropy = tree.entropy()
        self.assertEqual(entropy, 1.0)



class VectorEntropyTests(unittest.TestCase):

    def testZeroEntropy(self):
        for i in range(10):
            zeros = np.zeros(i)
            entropy = worldmodel.WorldModelTree._entropy(dist=zeros, normalize=True, ignore_empty_classes=True)
            self.assertEqual(entropy, 1.0)
        for i in range(2,10):
            zeros = np.zeros(i)
            entropy = worldmodel.WorldModelTree._entropy(dist=zeros, normalize=False, ignore_empty_classes=True)
            self.assertEqual(entropy, np.log2(i))

    def testOneEntropy(self):
        for i in range(2,10):
            ones = np.ones(i)
            entropy = worldmodel.WorldModelTree._entropy(dist=ones, normalize=True)
            self.assertEqual(entropy, 1.0)
        for i in range(2,10):
            ones = np.ones(i)
            entropy = worldmodel.WorldModelTree._entropy(dist=ones, normalize=False)
            self.assertEqual(entropy, np.log2(i))

    def testCustomEntropy(self):

        # normalized
        entropy = worldmodel.WorldModelTree._entropy(dist=np.array(range(2)), normalize=True)
        self.assertEqual(entropy, 0.0)
        entropy = worldmodel.WorldModelTree._entropy(dist=np.array(range(3)), normalize=True)
        self.assertEqual(entropy, 0.5793801642856950)
        entropy = worldmodel.WorldModelTree._entropy(dist=np.array(range(4)), normalize=True)
        self.assertEqual(entropy, 0.72957395851362239)

        # not normalized
        entropy = worldmodel.WorldModelTree._entropy(dist=np.array(range(2)), normalize=False)
        self.assertEqual(entropy, 0.0)
        entropy = worldmodel.WorldModelTree._entropy(dist=np.array(range(3)), normalize=False)
        self.assertEqual(entropy, 0.91829583405448956)
        entropy = worldmodel.WorldModelTree._entropy(dist=np.array(range(4)), normalize=False)
        self.assertEqual(entropy, 1.4591479170272448)

    def testRandomEntropy(self):
        """
        Checks wether entropy of random vectors stays between 0 and 1.
        """
        np.random.seed(0)
        for l in range(10):
            for _ in range(1000):
                p = l * np.random.random(l)
                e = worldmodel.WorldModelTree._entropy(dist=p, normalize=True)
                self.assertGreaterEqual(e, 0.)
                self.assertLessEqual(e, 1.)

        for l in range(2, 10):
            for _ in range(1000):
                p = l * np.random.random(l)
                e = worldmodel.WorldModelTree._entropy(dist=p, normalize=True)
                self.assertGreaterEqual(e, 0.)
                self.assertLessEqual(e, np.log2(l))


if __name__ == "__main__":
    unittest.main()
