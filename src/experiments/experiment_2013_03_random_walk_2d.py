import numpy as np

from matplotlib import pyplot

import worldmodel


class RandomWalk2DData(object):
    
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
        self.actions = []
        
        # generate the data
        self.data[0] = np.random.random(2)
        for i in range(1, n):
            # action
            a = np.random.randint(2)
            self.actions.append(a)
            # new data point
            x = self.data[i-1]
            y = np.array(x)
            y[a] += .1 * np.random.randn()
            # bounds
            y[a] = 0 if y[a] < 0 else y[a] 
            y[a] = 1 if y[a] > 1 else y[a]
            # store data
            self.data[i] = y
            
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
    data = RandomWalk2DData(n=4000)
    model = worldmodel.WorldModel(method='future')
    model.add_data(data=data.data, actions=data.actions)
    model.learn(min_gain=0.02)
    
    # plot data and result
    pyplot.subplot(1, 2, 1)
    data.plot(show_plot=False)
    pyplot.subplot(1, 2, 2)
    model.plot_states(show_plot=False)
    pyplot.show()
    