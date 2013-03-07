import numpy as np

from matplotlib import pyplot

import worldmodel


class NoisyDimData(object):
    
    def __init__(self, n=1000, seed=None):
        
        # store parameters
        self.n = n
        self.seed = seed
        self.actions = None
        
        # init random
        if seed is not None:
            np.random.seed(seed)
            
        # containers for data
        self.data = np.zeros((n, 2))
        self.labels = np.zeros(n, dtype=int)
        
        # generate the data
        self.data[0] = np.random.random(2)
        for i in range(1, n):
            # new data point
            x = self.data[i-1]
            y = np.array(x)
            y[0] += .1 * np.random.randn() # left-right step
            y[1] = np.random.random()      # complete noise
            # bounds
            y[0] = 0 if y[0] < 0 else y[0] 
            y[0] = 1 if y[0] > 1 else y[0]
            # store data
            self.data[i] = y
            
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
    data = NoisyDimData(n=5000)
    model = worldmodel.WorldModelTree()
    model.add_data(x=data.data, actions=None)
    model.learn(min_gain=0.02, max_costs=.02)
    
    # plot data and result
    pyplot.subplot(1, 2, 1)
    data.plot(show_plot=False)
    pyplot.subplot(1, 2, 2)
    model.plot_states(show_plot=False)
    pyplot.show()
    