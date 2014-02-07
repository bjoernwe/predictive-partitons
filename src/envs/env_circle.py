import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvCircle(environment.Environment):
    """A simple environment in which the agent moves along a circle.
    
    There is are two actions: PHI adds noise to the angle of the current 
    position and RAD adds noise to the radius.
    """

    def __init__(self, sigma_phi=0.1, sigma_rad=0.1, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        sigma_phi:   float - standard deviation for angular changes
        sigma_rad:   float - standard deviation for radial changes
        seed:        int - 
        """
        super(EnvCircle, self).__init__(seed=seed)
        
        self.ndim = 2
        self.sigma_phi = sigma_phi
        self.sigma_rad = sigma_rad
        self.actions_dict = {0: 'PHI', 1: 'RAD'}
        
        self.phi = 0
        self.rad = .5

        self.current_state = self._render(self.phi, self.rad)
        return
    
    
    def _render(self, phi, rad):
        
        a = np.cos(phi)
        b = np.sin(phi)
        R = np.array([[a, -b], [b, a]])
        
        e = rad * np.array([[1], [0]])
        return R.dot(e)[:,0]
    
    
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
            self.phi += self.rnd.normal(loc=0.0, scale=self.sigma_phi)
        elif action == 1:
            self.rad += self.rnd.normal(loc=0.0, scale=self.sigma_rad)
        else:
            assert False

        # stay in bounds            
        self.phi = self.phi % 2 * np.pi
        self.rad = 0 if self.rad < 0 else self.rad
        self.rad = 1 if self.rad > 1 else self.rad

        self.current_state = self._render(phi=self.phi, rad=self.rad)
        return self.current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 1000
    circle = EnvCircle()
    data, actions, _ = circle.do_random_steps(num_steps=steps)
    
    print 'Possible actions:'
    for action, describtion in circle.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
        
    # plot data
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()
    