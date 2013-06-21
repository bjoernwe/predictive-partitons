import numpy as np

from matplotlib import pyplot

import worldmodel


class RandomSwissRollData(object):
    
    def __init__(self, n=1000, seed=None):
        
        # store parameters
        self.n = n
        self.seed = seed
        
        # init random
        if seed is not None:
            np.random.seed(seed)
            
        # containers for data
        self.data = np.zeros((n, 2))
        self.labels = np.zeros(n, dtype=int)
        self.actions = None
        
        # generate the data
        fourpi = 4. * np.pi
        t = fourpi / 2.
        for i in range(0, n):
            # random walk
            t += .5 * np.random.randn()
            # bounds
            t = 0 if t < 0 else t 
            t = fourpi if t > fourpi else t
            # data point
            x = np.cos(t)*(1-.7*t/fourpi)
            y = np.sin(t)*(1-.7*t/fourpi)
            self.data[i,0] = x
            self.data[i,1] = y
            
        return
    
    
    def plot(self, show_plot=True):

        n = self.data.shape[0]
        
        for i in range(n-1):
            pyplot.plot(self.data[i:i+2,0], self.data[i:i+2,1], 'b')

        if show_plot:            
            pyplot.show()
            
        return


            
if __name__ == '__main__':

    # train model
    data = RandomSwissRollData(n=1000)
    model = worldmodel.WorldModelSpectral()
    model.add_data(x=data.data, actions=data.actions)
    model.learn(min_gain=0.02)
    
    # plot data and result
    pyplot.subplot(1, 2, 1)
    data.plot(show_plot=False)
    pyplot.subplot(1, 2, 2)
    model.plot_states(show_plot=False)
    model.plot_states(show_plot=False)
    model.plot_tree_data(color='state', show_plot=False)
    pyplot.show()
    