import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvCube(environment.Environment):
    """A simple environment in which the agent moves inside a (hyper-) cube 
    between zero and one.
    
    At each step in every dimension some Gaussian noise is added with standard 
    deviation sigma.
    """

    def __init__(self, sigma=0.1, ndim=2, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        sigma:       float - standard deviation
        ndim:        int - dimensionality of the generated cube
        seed:        int - 
        """
        super(EnvCube, self).__init__(seed=seed)
        self.ndim = ndim
        self.sigma = sigma
        self.current_state = .5 * np.ones(ndim)
        return
    
    
    def _do_action(self, action):
        """Perform a random step and return resulting state of the environment 
        as well as the reward (in this case 0).
        --------------------------------------
        Parameters:
        action:     0
        --------------------------------------
        Return:
        new_state:    np.ndarray - coordinates of the agent after the step
        reward = 0
        """
        
        # perform step
        for d in range(self.ndim):
            self.current_state[d] += self.sigma * self.rnd.normal()
        self.current_state = np.clip(self.current_state, 0, 1)
        return self.current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 1000
    cube = EnvCube()
    data = cube.do_random_steps(num_steps=steps)[0]
    
    print 'Possible actions:'
    for action, describtion in cube.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    # plot data
    plt.scatter(data[:,0], data[:,1])
    plt.show()
    