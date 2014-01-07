import numpy as np
import unittest

import entropy_utils


class VectorEntropyTests(unittest.TestCase):


    def testZeroEntropy(self):
        for i in range(10):
            zeros = np.zeros(i)
            entropy = entropy_utils.entropy(zeros, normalize=True, ignore_empty_classes=True)
            self.assertEqual(entropy, 1.0)
        for i in range(2,10):
            zeros = np.zeros(i)
            entropy = entropy_utils.entropy(zeros, normalize=False, ignore_empty_classes=True)
            self.assertEqual(entropy, np.log2(i))


    def testOneEntropy(self):
        for i in range(2,10):
            ones = np.ones(i)
            entropy = entropy_utils.entropy(ones, normalize=True)
            self.assertAlmostEqual(entropy, 1.0, 6)
        for i in range(2,10):
            ones = np.ones(i)
            entropy = entropy_utils.entropy(ones, normalize=False)
            self.assertAlmostEqual(entropy, np.log2(i), 6)


    def testCustomEntropy(self):

        # normalized
        entropy = entropy_utils.entropy(np.array(range(2)), normalize=True)
        self.assertEqual(entropy, 0.0)
        entropy = entropy_utils.entropy(np.array(range(3)), normalize=True)
        self.assertAlmostEqual(entropy, 0.5793801642856950, 6)
        entropy = entropy_utils.entropy(np.array(range(4)), normalize=True)
        self.assertAlmostEqual(entropy, 0.7295739585136223, 6)

        # not normalized
        entropy = entropy_utils.entropy(np.array(range(2)), normalize=False)
        self.assertEqual(entropy, 0.0)
        entropy = entropy_utils.entropy(np.array(range(3)), normalize=False)
        self.assertAlmostEqual(entropy, 0.9182958340544895, 6)
        entropy = entropy_utils.entropy(np.array(range(4)), normalize=False)
        self.assertAlmostEqual(entropy, 1.4591479170272448, 6)


    def testRandomEntropy(self):
        """
        Checks whether entropy of random vectors stays between 0 and 1.
        """
        np.random.seed(0)
        for l in range(10):
            for _ in range(1000):
                p = l * np.random.random(l)
                e = entropy_utils.entropy(p, normalize=True)
                self.assertGreaterEqual(e, 0.)
                self.assertLessEqual(e, 1.)

        for l in range(2, 10):
            for _ in range(1000):
                p = l * np.random.random(l)
                e = entropy_utils.entropy(p, normalize=True)
                self.assertGreaterEqual(e, 0.)
                self.assertLessEqual(e, np.log2(l))
                
                
    def testEntropyRate(self):
        """
        Tests a few hand-crafted entropy rates.
        """
        
        # zeros
        P = np.zeros((4, 4))
        mu = np.array([10, 0, 0, 0])
        self.failUnlessAlmostEqual(entropy_utils.entropy_rate(P, mu, normalize=False), 2.0)
        
        # zero entropy
        P = np.eye(4) 
        self.failUnlessAlmostEqual(entropy_utils.entropy_rate(P, mu, normalize=False), 0)
        
        # ones
        for i in range(2,10):
            P = np.ones((i, i))
            mu = np.ones(i)
            self.failUnlessAlmostEqual(entropy_utils.entropy_rate(P, mu, normalize=False), np.log2(i), 6)
            
        # custom
        P = np.array([[1, 2], [3, 4]])
        mu = np.array([1, 0])
        self.failUnlessAlmostEqual(entropy_utils.entropy_rate(P, mu, normalize=False), 0.91829574108123779, 6)        
        mu = np.array([0, 1])
        self.failUnlessAlmostEqual(entropy_utils.entropy_rate(P, mu, normalize=False), 0.98522818088531494, 6)        
        mu = np.array([1., 1.])
        self.failUnlessAlmostEqual(entropy_utils.entropy_rate(P, mu, normalize=False), 0.95176196098327637, 6)        
        self.failUnlessAlmostEqual(entropy_utils.entropy_rate(P, mu=None, normalize=False), 0.95903722276316206, 6)
        
        P = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])        
        self.failUnlessAlmostEqual(entropy_utils.entropy_rate(P, mu=None, normalize=False), 1.5436952667895776, 6)
        
        return
        

    def testMutualInformation(self):                
        """
        Calculates Mutual Information for some hand-crafted matrixes.
        """
        
        # zero entropy
        P = np.eye(4) 
        self.failUnlessAlmostEqual(entropy_utils.mutual_information(P), 2)
        P = np.array([[1, 2], [3, 4]])
        self.failUnlessAlmostEqual(entropy_utils.mutual_information(P), 0.0065989113844819869)
        P = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])        
        self.failUnlessAlmostEqual(entropy_utils.mutual_information(P), 0.012238804751254495)
        
        return


if __name__ == "__main__":
    unittest.main()