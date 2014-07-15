import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvRibbon(environment.Environment):
    """A simple environment in which the agent moves along a env."""

    def __init__(self, step_size=1, seed=None):
        super(EnvRibbon, self).__init__(seed=seed)
        
        self.ndim = 2
        self.step_size = step_size
        self.actions_dict = {0: 'NONE'}
        
        self.phi = 0
        self.current_state = self._render(self.phi)
        return
    
    
    def _render(self, phi):
        x = np.cos(phi)
        y = np.sin(2*phi)
        return np.array([x,y])
        
    
    def _do_action(self, action):
        """Walks one step along the env.
        Returns new state and new angle.
        """
        
        if action == 0:
            self.phi += self.step_size
            self.phi = self.phi % (2 * np.pi)
        else:
            assert False

        self.current_state = self._render(self.phi)
        return self.current_state, self.phi


if __name__ == '__main__':
    
    # sample data
    steps = 1000
    env = EnvRibbon(step_size=1)
    data, actions, _ = env.do_random_steps(num_steps=steps)
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
        
    # plot data
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()
    