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
        tree = worldmodel.WorldModelTree(normalized_entropy=True, global_entropy='weighted')
        tree.add_data(data)
        trans = tree.transitions
        self.assertEqual(trans.shape, (1,1))
        self.assertEqual(trans[0,0], data.shape[0]-1) # n of transitions

        

class TrivialEntropyTests(unittest.TestCase):

    def setUp(self):
        self.data = worldmodel.problemChain(n=1000, seed=0)

    def testTrivialEntropy(self):
        for normalized_entropy in [False, True]:
            for global_entropy in ['sum', 'avg', 'weighted']:
                tree = worldmodel.WorldModelTree(normalized_entropy=normalized_entropy, global_entropy=global_entropy)
                tree.add_data(self.data)
                entropy = tree.entropy()
                self.assertEqual(entropy, 1.0)

    def testTrivialTransEntropy(self):
        for normalized_entropy in [False, True]:
            for global_entropy in ['sum', 'avg', 'weighted']:
                tree = worldmodel.WorldModelTree(normalized_entropy=normalized_entropy, global_entropy=global_entropy)
                tree.add_data(self.data)
                trans = tree.transitions
                entropy = tree._transition_entropy(trans_matrix=trans)
                self.assertEqual(entropy, 1.0)

    def testTrivialLeafEntropy(self):
        for normalized_entropy in [False, True]:
            for global_entropy in ['sum', 'avg', 'weighted']:
                tree = worldmodel.WorldModelTree(normalized_entropy=normalized_entropy, global_entropy=global_entropy)
                tree.add_data(self.data)
                entropy = tree._entropy()
                self.assertEqual(entropy, 1.0)



class VectorEntropyTests(unittest.TestCase):

    def testZeroEntropy(self):
        for normalized_entropy in [False, True]:
            for global_entropy in ['sum', 'avg', 'weighted']:
                tree = worldmodel.WorldModelTree(normalized_entropy=normalized_entropy, global_entropy=global_entropy)
                for i in range(10):
                    zeros = np.zeros(i)
                    entropy = tree._entropy(trans=zeros, ignore_empty_classes=True)
                    self.assertEqual(entropy, 1.0)

    def testOneEntropy(self):
        for global_entropy in ['sum', 'avg', 'weighted']:
            tree = worldmodel.WorldModelTree(normalized_entropy=True, global_entropy=global_entropy)
            for i in range(2,10):
                ones = np.ones(i)
                entropy = tree._entropy(trans=ones)
                self.assertEqual(entropy, 1.0)
            tree = worldmodel.WorldModelTree(normalized_entropy=False, global_entropy=global_entropy)
            for i in range(2,10):
                ones = np.ones(i)
                entropy = tree._entropy(trans=ones)
                self.assertEqual(entropy, np.log2(i))

    def testCustomEntropy(self):

        for global_entropy in ['sum', 'avg', 'weighted']:

            # normalized
            tree = worldmodel.WorldModelTree(normalized_entropy=True, global_entropy=global_entropy)
            entropy = tree._entropy(trans=np.array(range(2)))
            self.assertEqual(entropy, 0.0)
            entropy = tree._entropy(trans=np.array(range(3)))
            self.assertEqual(entropy, 0.5793801642856950)
            entropy = tree._entropy(trans=np.array(range(4)))
            self.assertEqual(entropy, 0.72957395851362239)

            # not normalized
            tree = worldmodel.WorldModelTree(normalized_entropy=False, global_entropy=global_entropy)
            entropy = tree._entropy(trans=np.array(range(2)))
            self.assertEqual(entropy, 0.0)
            entropy = tree._entropy(trans=np.array(range(3)))
            self.assertEqual(entropy, 0.91829583405448956)
            entropy = tree._entropy(trans=np.array(range(4)))
            self.assertEqual(entropy, 1.4591479170272448)

    def testRandomEntropy(self):
        """
        Checks wether entropy of random vectors stays between 0 and 1.
        """
        np.random.seed(0)
        tree = worldmodel.WorldModelTree(normalized_entropy=True, global_entropy='weighted')
        for l in range(10):
            for _ in range(1000):
                p = l * np.random.random(l)
                e = tree._entropy(trans=p)
                self.assertGreaterEqual(e, 0.)
                self.assertLessEqual(e, 1.)

        tree = worldmodel.WorldModelTree(normalized_entropy=False, global_entropy='weighted')
        for l in range(2, 10):
            for _ in range(1000):
                p = l * np.random.random(l)
                e = tree._entropy(trans=p)
                self.assertGreaterEqual(e, 0.)
                self.assertLessEqual(e, np.log2(l))


if __name__ == "__main__":
    unittest.main()
