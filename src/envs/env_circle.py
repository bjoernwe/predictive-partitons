import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvCircle(environment.Environment):
    """A simple environment in which the agent moves along a circle."""

    def __init__(self, step_size=2, seed=None):
        super(EnvCircle, self).__init__(seed=seed)
        
        self.ndim = 2
        self.step_size = step_size
        self.actions_dict = {0: 'NONE'}
        
        self.phi = 0
        self.current_state = self._render(self.phi)
        return
    
    
    def _render(self, phi):
        x = np.cos(phi)
        y = np.sin(phi)
        return np.array([x,y])
        
    
    def _do_action(self, action):
        """Perform the given action and return the resulting state of the
        environment and the action as well.
        --------------------------------------
        Parameters:
        action:     int
        --------------------------------------
        Return:
        new_state:    np.ndarray - coordinates of the agent after the step
        reward = 0
        """
        
        if action == 0:
            self.phi += self.step_size
            self.phi = self.phi % (2 * np.pi)
        else:
            assert False

        self.current_state = self._render(self.phi)
        return self.current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 100
    circle = EnvCircle()
    data, actions, _ = circle.do_random_steps(num_steps=steps)
    
    print 'Possible actions:'
    for action, describtion in circle.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
        
    # plot data
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()
    