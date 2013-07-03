"""
In this experiment, a random walk in 2D is performed. One dimension is noisy but
the strength of the noise varies along the axis to show that the partition
sizes also change depending on how predictable the environment behaves.
"""

import numpy as np

from matplotlib import pyplot

import worldmodel


class SimpleActionsData(object):
    
    def __init__(self, n=1000, seed=None):
        
        # store parameters
        self.n = n
        self.seed = seed
        
        # init random
        if seed is not None:
            np.random.seed(seed)
            
        # containers for data
        self.data_true = np.zeros((n, 2))
        self.data = np.zeros((n, 2))
        self.labels = np.zeros(n, dtype=int)
        self.actions = []
        
        # generate the data
        self.data_true[0] = np.random.random(2)
        self.data[0] = self.data_true[0]
        for i in range(1, n):
            # action
            a = np.random.randint(2)
            self.actions.append(a)
            # new data point
            x = self.data_true[i-1]
            y = np.array(x)
            y[a] += .1 * np.random.randn()
            # bounds
            y[a] = 0 if y[a] < 0 else y[a] 
            y[a] = 1 if y[a] > 1 else y[a]
            # add noise to first dimension depending on position
            noise_factor = y[1]
            z = np.array(y)
            z[0] = (1. - noise_factor) * y[0] + noise_factor * np.random.random()
            # store data
            self.data_true[i] = y
            self.data[i] = z
            
        return
    
    
    def plot(self, show_plot=True):

        n = self.data.shape[0]
        
        for i in range(n-1):
            color = 'r' if self.actions[i] else 'b'
            pyplot.plot(self.data[i:i+2,0], self.data[i:i+2,1], color)

        if show_plot:            
            pyplot.show()
            
        return

            
if __name__ == '__main__':

    # train model
    data = SimpleActionsData(n=5000)
    model = worldmodel.WorldModelSpectral()
    model.add_data(x=data.data, actions=data.actions)
    model.learn(min_gain=0.02)
    
    # plot data and result
    pyplot.subplot(1, 2, 1)
    data.plot(show_plot=False)
    pyplot.subplot(1, 2, 2)
    model.plot_states(show_plot=False)
    pyplot.show()
    