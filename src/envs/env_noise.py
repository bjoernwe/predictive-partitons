import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvNoise(environment.Environment):
    '''
    A simple environment in which the agent performs a random walk in one 
    dimension while the others are completely noisy'''

    def __init__(self, sigma=0.1, ndim=2, seed=None):
        """
        Initialize the environment.
        """
        super(EnvNoise, self).__init__(seed=seed)
        assert ndim >= 2
        self.ndim = ndim
        self.current_state = np.zeros(ndim)
        self.actions = None
        self.sigma = sigma
        return
    
    
    def _do_action(self, action):
        """
        Performs the given action and returns the resulting state of the
        environment as well as zero reward.
        """
        
        # perform step in dim 0
        x = self.current_state[0] 
        x += self.rnd.normal(loc=0, scale=self.sigma)
        
        # stay in cube
        x = 0 if x < 0 else x
        x = 1 if x > 1 else x
        self.current_state[0] = x
        
        # noise in other dims
        for i in range(1, self.ndim):
            self.current_state[i] = self.rnd.uniform()
        
        return self.current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 1000
    cube = EnvNoise()
    data = cube.do_random_steps(num_steps=steps)[0]
    
    # plot data
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()
    