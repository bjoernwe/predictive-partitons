import numpy as np
import random

from matplotlib import pyplot

import worldmodel

import sierpinsky


class VoronoiData(object):
    
    symbols = ['o', '^', 'd', 's', '*']
    
    def __init__(self, n=1000, k=3, power=1, seed=None):
        
        # store parameters
        self.n = n
        self.k = k
        self.power = power
        self.seed = seed
        
        # init random
        if seed is not None:
            np.random.seed(seed)
    
        # k class means
        self.means = np.random.random((k, 2))
        
        ### init transition probabilities ###
        
        ### deterministic transitions
        #probs = np.zeros((k,k))
        #for i in range(k):
        #    probs[i,(i+1)%k] = 1
            
        ### fifty, fifty
        probs = np.zeros((k,k))
        for i in range(k):
            probs[i,i] = 1
            probs[i,(i+1)%k] = .5
            probs[i,(i+2)%k] = .25
            #probs[i,(i+3)%k] = .125
            #probs[i,(i+4)%k] = .0625
        
        # transition probabilities between classes
        #probs = np.random.random((k, k))**power
        
        # sierpinsky
        #probs = sierpinsky.sierpinsky_square(N=k)
        
        # normalize probabilities
        probs = probs / np.sum(probs, axis=1)[:,np.newaxis]
        self.probs = probs
        
        # generate the actual data
        self.data = np.zeros((n, 2))
        self.labels = np.zeros(n, dtype=int)
        current_class = 0
        for i in range(n):
            
            if not i%1000:
                print i
        
            # weighted sample of class means
            # from: http://stackoverflow.com/questions/6432499/how-to-do-weighted-random-sample-of-categories-in-python
            next_class = np.array(probs[current_class]).cumsum().searchsorted(np.random.sample(1))[0]
            
            closest_class = -1
            while not (closest_class == next_class):
                x = np.random.random(2)
                closest_class = self.classify(x)
                
            self.data[i] = x
            self.labels[i]= closest_class
            current_class = next_class
    
        # calculate transition matrix
        self.transitions = np.zeros((k,k))
        prev_class = self.labels[0]
        for i in range(self.data.shape[0]-1):
            next_class = self.labels[i+1]
            self.transitions[prev_class, next_class] += 1
            prev_class = next_class
            
        return
    
        
    def classify(self, x):
        distances = map(lambda m: np.linalg.norm(x-m), self.means)
        closest_class = np.argmin(distances)
        return closest_class
    
    
    def plot(self, show_plot=True):

        # each class into one list        
        data_list = [[] for _ in range(self.k)]
        for i, label in enumerate(self.labels):
            data_list[label].append(self.data[i])
        for i in range(self.k):
            data_list[i] = np.vstack(data_list[i])
            
        # plot
        colormap = pyplot.cm.prism
        pyplot.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.98, 7)])
        for i, data in enumerate(data_list):
            pyplot.plot(data[:,0], data[:,1], self.symbols[i%len(self.symbols)])

        if show_plot:            
            pyplot.show()
        return
    
    
    def entropy(self, normalize=False):
        return worldmodel.WorldModelTree._matrix_entropy(transitions=self.transitions, normalize=normalize)


if __name__ == '__main__':
    
    k = 8
    voronoi = VoronoiData(n=10000, k=k, power=5)
    entropy = voronoi.entropy() 
    print voronoi.transitions
    print 'eigenvalues:\n', np.abs(np.linalg.eig(voronoi.transitions)[0])
    
    model = worldmodel.WorldModelTree()
    model.add_data(voronoi.data)
    model.learn(min_gain=0.02, max_costs=0.02)
    print 'final number of nodes:', len(model._nodes())
    
    # plot target
    pyplot.subplot(2,2,1)
    voronoi.plot(show_plot=False)
     
    # plot result
    pyplot.subplot(2,2,2)
    model.plot_tree_data(show_plot=False)
    
    # plot stats
    pyplot.subplot(2,1,2)
    model.plot_stats(show_plot=False)
    pyplot.show()
    