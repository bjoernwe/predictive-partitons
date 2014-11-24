import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvSine(environment.Environment):
    """A simple environment in which generates a sine wave.
    """

    def __init__(self, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:        int - 
        """
        super(EnvSine, self).__init__(seed=seed)
        self.ndim = 1
        self.counter = 0
        self.current_state = np.sin(self.counter)
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
        self.current_state = np.sin(self.counter)
        return self.current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 10
    env = EnvSine()
    data = env.do_random_steps(num_steps=steps)[0]
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    # plot data
    plt.plot(data)
    plt.show()
    