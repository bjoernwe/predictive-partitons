# coding: latin-1

import numpy as np

from matplotlib import pyplot

import worldmodel


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
            #probs[i,(i+2)%k] = .25
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
            if len(data_list[i]) > 0:
                data_list[i] = np.vstack(data_list[i])
            else:
                data_list[i] = None
            
        # plot
        colormap = pyplot.cm.prism
        pyplot.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.98, 7)])
        for i, data in enumerate(data_list):
            if data is not None:
                pyplot.plot(data[:,0], data[:,1], self.symbols[i%len(self.symbols)])

        if show_plot:            
            pyplot.show()
        return
    
    
    def entropy(self, normalize=False):
        return worldmodel.WorldModelTree._matrix_entropy(transitions=self.transitions, normalize=normalize)


def experiment_plot():
    
    k = 8
    voronoi = VoronoiData(n=2000, k=k, power=5)
    #print voronoi.transitions
    #print 'eigenvalues:\n', np.abs(np.linalg.eig(voronoi.transitions)[0])
    
    model = worldmodel.WorldModelTree()
    model.add_data(voronoi.data)
    #model.learn(min_gain=0.015, max_costs=0.015)
    model.single_splitting_step()
    model.single_splitting_step()
    #model.single_splitting_step()
    #model.single_splitting_step()
    #model.single_splitting_step()
    
    print ''
    print 'final number of nodes:', len(model._nodes())
    print 'mutual information (model):', model._mutual_information(voronoi.probs)
    print 'mutual information (data):', model._mutual_information(voronoi.transitions)
    print 'mutual information (learned model):', model._mutual_information(model.transitions)
    
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
    
    
def experiment_average():

    N = 100
    n_data_points = 5000
    
    stats_n_nodes = np.zeros(N)
    stats_mi_model = np.zeros(N)
    stats_mi_data = np.zeros(N)
    stats_mi_learned = np.zeros(N)
    stats_mi_diff = np.zeros(N)
    
    for i in range(N):
        
        voronoi = VoronoiData(n=n_data_points, k=8, power=5)
        model = worldmodel.WorldModelTree()
        model.add_data(voronoi.data)
        model.learn(min_gain=0.015, max_costs=0.015)
        
        stats_n_nodes[i] = len(model._nodes())
        stats_mi_model[i] = model._mutual_information(transition_matrix=voronoi.probs)
        stats_mi_data[i] = model._mutual_information(transition_matrix=voronoi.transitions)
        stats_mi_learned[i] = model._mutual_information(transition_matrix=model.transitions)
        stats_mi_diff[i] = abs(stats_mi_model[i] - stats_mi_learned[i])
        
    print '# of trials:', N
    print '# data points:', n_data_points
    print '# of nodes: {avg:.2f} ± {std:.2f}'.format(avg=np.average(stats_n_nodes), std=np.std(stats_n_nodes))
    print 'mutual information (model):   {avg:.2f} ± {std:.2f}'.format(avg=np.average(stats_mi_model), std=np.std(stats_mi_model))
    print 'mutual information (data):    {avg:.2f} ± {std:.2f}'.format(avg=np.average(stats_mi_data), std=np.std(stats_mi_data))
    print 'mutual information (learned): {avg:.2f} ± {std:.2f}'.format(avg=np.average(stats_mi_learned), std=np.std(stats_mi_learned))
    print 'mutual information (error):   {avg:.2f} ± {std:.2f}'.format(avg=np.average(stats_mi_diff), std=np.std(stats_mi_diff))


if __name__ == '__main__':

    experiment_plot()
    #experiment_average()
    