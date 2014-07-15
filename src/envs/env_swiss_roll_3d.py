import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import environment


class EnvSwissRoll3D(environment.Environment):

    def __init__(self, sigma_phi=0.5, sigma_z=0.05, seed=None):
        """
        Initializes the environment including an initial state.
        """
        super(EnvSwissRoll3D, self).__init__(seed=seed)
            
        self.ndim = 3
        self.actions = None
        
        self.sigma_phi = sigma_phi
        self.sigma_z = sigma_z
        self.fourpi = 4. * np.pi
        
        self.phi = self.fourpi / 2.
        self.z = 0.5
        self.current_state = np.hstack([self._f(self.phi), self.z])
        return
        
        
    def _f(self, phi):
        """
        Maps an angle phi to x, y values of the swiss roll.
        """
        x = np.cos(phi)*(1-.7*phi/self.fourpi)
        y = np.sin(phi)*(1-.7*phi/self.fourpi)
        return np.array([x, y])
        

    def _do_action(self, action):
        """
        Performs an random step on the swiss roll and returns the new data value
        along with a reward/label value (in this case the angle on the spiral).
        """

        # random walk
        self.phi += self.sigma_phi * self.rnd.normal()
        self.z += self.sigma_z * self.rnd.normal()
        
        # bounds
        self.phi = np.clip(self.phi, 0, self.fourpi)
        self.z = np.clip(self.z, 0, 1)
        
        # result
        self.current_state = np.hstack([self._f(self.phi), self.z])
        return self.current_state, self.phi



if __name__ == '__main__':

    steps = 3000
    env = EnvSwissRoll3D()
    data, _, labels = env.do_random_steps(num_steps=steps)
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
        
    # plot data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data[:-1,0], data[:-1,2], data[:-1,1], c=labels)
    plt.show()
    