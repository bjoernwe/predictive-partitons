import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvCube(environment.Environment):
    """A simple environment in which the agent moves inside a (hyper-) cube 
    between zero and one.
    
    There is one UP and one DOWN action for each dimension which performs a
    step of given length. To each step some Gaussian noise is added with 
    standard deviation sigma.
    """

    def __init__(self, step_size=0.1, sigma=0.1, ndim=2, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        step_size:   float
        sigma:       float - standard deviation
        ndim:        int - dimensionality of the generated cube
        seed:        int - 
        """
        super(EnvCube, self).__init__(seed=seed)
        self.ndim = ndim
        self.current_state = np.zeros(ndim)
        
        self.step_size = step_size
        self.sigma = sigma
        
        # initialize actions (UP/DOWN for each dimension)
        self.actions_dict = {}
        for i in range(ndim):
            self.actions_dict[2*i]   = ('D%d' % i)
            self.actions_dict[2*i+1] = ('U%d' % i)
            
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
        
        # auxiliary variables
        dim = action // 2
        direction = -1 if action % 2 == 0 else 1
        
        # perform step
        x = self.current_state[dim]
        noise = self.rnd.normal(loc=0, scale=self.sigma)
        x += (direction * self.step_size) + noise
        
        # stay in cube
        x = 0 if x < 0 else x
        x = 1 if x > 1 else x
        self.current_state[dim] = x
        
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
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()
    