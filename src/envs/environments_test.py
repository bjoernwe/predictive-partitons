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

        #print 'Testing environments with seed:'
        for Env in self.environments:
            
            #print Env.__name__
            
            env = Env(seed=3)
            A, actions_1, rewards_1 = env.generate_training_data(num_steps=100, noisy_dims=2)[0]
            
            env = Env(seed=3)
            B, actions_2, rewards_2 = env.generate_training_data(num_steps=100, noisy_dims=2)[0]
            
            assert np.array_equal(A, B)
            assert np.array_equal(actions_1, actions_2)
            assert np.array_equal(rewards_1, rewards_2)
        
        #print 'Done.\n'
        return


    def testSeedMissing(self):
        """
        Tests whether the result is random when seed is missing.
        """

        #print 'Testing environments without seed:'
        for Env in self.environments:
        
            #print Env.__name__
            
            env = Env()
            A, actions_1, _ = env.generate_training_data(num_steps=100, noisy_dims=2)[0]
            
            env = Env()
            B, actions_2, _ = env.generate_training_data(num_steps=100, noisy_dims=2)[0]
        
            assert not np.array_equal(A, B)
            if env.get_number_of_possible_actions() >= 2:
                assert not np.array_equal(actions_1, actions_2)
            
        #print 'Done.\n'
        return 


if __name__ == "__main__":
    unittest.main()
    
