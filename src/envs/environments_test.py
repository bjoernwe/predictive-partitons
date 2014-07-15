import numpy as np
import unittest

import env_cube
import env_disk
import env_noise
import env_oscillator
import env_ribbon
import env_swiss_roll
import env_swiss_roll_3d


class EnvironmentTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.environments = []
        self.environments.append(env_cube.EnvCube)    
        self.environments.append(env_disk.EnvDisk)    
        self.environments.append(env_noise.EnvNoise)    
        self.environments.append(env_oscillator.EnvOscillator)    
        self.environments.append(env_ribbon.EnvRibbon)    
        self.environments.append(env_swiss_roll.EnvSwissRoll)    
        self.environments.append(env_swiss_roll_3d.EnvSwissRoll3D)    
              

    def testSeedGiven(self):
        """
        Tests whether the result depends on the seed.
        """
        for Env in self.environments:
            env = Env(seed=3)
            A, actions_1, rewards_1 = env.do_random_steps(num_steps=100)
            
            env = Env(seed=3)
            B, actions_2, rewards_2 = env.do_random_steps(num_steps=100)
            
            assert np.array_equal(A, B)
            assert np.array_equal(actions_1, actions_2)
            assert np.array_equal(rewards_1, rewards_2)
        
        return


    def testSeedMissing(self):
        """
        Tests whether the result is random when seed is missing.
        """
        for Env in self.environments:
        
            env = Env()
            A, actions_1, _ = env.do_random_steps(num_steps=100)
            
            env = Env()
            B, actions_2, _ = env.do_random_steps(num_steps=100)
        
            assert not np.array_equal(A, B)
            if env.get_number_of_possible_actions() >= 2:
                assert not np.array_equal(actions_1, actions_2)
            
        return 


if __name__ == "__main__":
    unittest.main()
    
