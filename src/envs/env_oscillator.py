import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvOscillator(environment.Environment):
    """A simple environment in which the system's state oscillates between the
    corners of a unit (hyper-) cube.
    """

    def __init__(self, ndim=2, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        ndim:        int - dimensionality of the generated cube
        seed:        int - 
        """
        super(EnvOscillator, self).__init__(seed=seed)
        self.ndim = ndim
        self.noisy_dim_dist = 'binary'
        self.current_state = np.zeros(ndim)
        self.counter = 0
        return
    
    
    def _do_action(self, action):
        """Perform the given action and return the resulting state of the
        environment and the reward as well.
        --------------------------------------
        Parameters:
        action:     str - direction of the action to be performed
        --------------------------------------
        Return:
        new_state:    np.ndarray - coordinates of the agent after the step
        reward = 0
        """
        
        self.counter += 1
        
        for i in range(self.ndim):
            self.current_state[i] = (self.counter >> i) % 2
        
        return self.current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 10
    cube = EnvOscillator()
    data = cube.do_random_steps(num_steps=steps)[0]
    
    print 'Possible actions:'
    for action, describtion in cube.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    # plot data
    plt.scatter(data[:,0], data[:,1])
    plt.show()
    