import numpy as np
import matplotlib.pyplot as plt

import environment


class EnvSwissRoll(environment.Environment):

    def __init__(self, sigma=0.5, seed=None):
        """
        Initializes the environment including an initial state.
        """
        super(EnvSwissRoll, self).__init__(seed=seed)
        #self.rnd = random.Random()
        #if seed is not None:
        #    self.rnd.seed(seed)
            
        self.ndim = 2
        self.actions = None
        
        self.sigma = sigma
        self.threepi = 4. * np.pi
        
        self.t = self.threepi / 2.
        self.current_state = self._f(self.t)
        
        
    def _f(self, phi):
        """
        Maps an angle phi to x, y values of the swiss roll.
        """
        x = np.cos(phi)*(1-.7*phi/self.threepi)
        y = np.sin(phi)*(1-.7*phi/self.threepi)
        return np.array([x, y])
        

    def _do_action(self, action):
        """
        Performs an random step on the swiss roll and returns the new data value
        along with a reward/label value (in this case the angle on the spiral).
        """

        # random walk
        #self.t += self.t * self.rnd.gauss(mu=0, sigma=1) / self.threepi
        self.t += self.sigma * self.rnd.normal()
        
        # bounds
        self.t = 0 if self.t < 0 else self.t 
        self.t = self.threepi if self.t > self.threepi else self.t
        
        # result
        self.current_state = self._f(self.t)
        return self.current_state, self.t



if __name__ == '__main__':

    steps = 1000
    env = EnvSwissRoll(sigma=0.5)
    data, actions, _ = env.do_random_steps(num_steps=steps)
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
        
    # plot data
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()
    