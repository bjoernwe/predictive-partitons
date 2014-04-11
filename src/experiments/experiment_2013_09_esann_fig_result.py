"""
Generates plots for the ESANN paper.
"""

import cPickle as pickle
import numpy as np

from matplotlib import pyplot

            
if __name__ == '__main__':
    
    #
    # parameters
    #
    T = 10
    resolution = 100


    #
    # load data
    #
    results = {}
    for t in range(T):
        results[t] = {}
        results[t]['models'] = pickle.load(open('experiment_2014_07_esann_models_%d.dump' % t, 'rb') )
        results[t]['data'] = pickle.load(open('experiment_2014_07_esann_data_%d.dump' % t, 'rb') )
        print len(results[t]['models']['swiss_predictive'].stats)
        print len(results[t]['models']['noise_predictive'].stats)
        print ''
        
    #
    # plot results swiss role 
    #
    N = 24
    pyplot.subplot(1, 2, 1)
    pyplot.xlabel('number of states')
    pyplot.ylabel('mutual information')
    pyplot.title('Problem A')

    results_mi = np.zeros((T, 4))
    results_nodes = np.zeros((T, 4))
    for t in range(T):
        model = results[t]['models']['swiss_naive']
        results_mi[t,:] = np.array([s.mutual_information for s in model.stats])[:4]
        results_nodes[t,:] = np.array([s.n_states for s in model.stats])[:4]
    n_states = np.mean(results_nodes, axis=0)
    mi_mean = np.mean(results_mi, axis=0)
    mi_std = np.std(results_mi, axis=0)
    pyplot.errorbar(x=n_states, y=mi_mean, yerr=mi_std, fmt='o-')
    
    results_mi = np.zeros((T, N))
    results_nodes = np.zeros((T, N))
    for t in range(T):
        model = results[t]['models']['swiss_predictive']
        results_mi[t,:] = np.array([s.mutual_information for s in model.stats])[:N]
        results_nodes[t,:] = np.array([s.n_states for s in model.stats])[:N]
    n_states = np.mean(results_nodes, axis=0)
    mi_mean = np.mean(results_mi, axis=0)
    mi_std = np.std(results_mi, axis=0)
    pyplot.errorbar(x=n_states, y=mi_mean, yerr=mi_std, fmt='-')

    pyplot.legend(['naive', 'predicitve'], loc='lower right')


    #
    # plot results noise 
    #
    N = 16
    pyplot.subplot(1, 2, 2)
    pyplot.xlabel('number of states')
    pyplot.title('Problem B')
    
    results_mi = np.zeros((T, 4))
    results_nodes = np.zeros((T, 4))
    for t in range(T):
        model = results[t]['models']['noise_naive']
        results_mi[t,:] = np.array([s.mutual_information for s in model.stats])[:4]
        results_nodes[t,:] = np.array([s.n_states for s in model.stats])[:4]
    n_states = np.mean(results_nodes, axis=0)
    mi_mean = np.mean(results_mi, axis=0)
    mi_std = np.std(results_mi, axis=0)
    pyplot.errorbar(x=n_states, y=mi_mean, yerr=mi_std, fmt='o-')

    results_mi = np.zeros((T, N))
    results_nodes = np.zeros((T, N))
    for t in range(T):
        model = results[t]['models']['noise_predictive']
        results_mi[t,:] = np.array([s.mutual_information for s in model.stats])[:N]
        results_nodes[t,:] = np.array([s.n_states for s in model.stats])[:N]
    n_states = np.mean(results_nodes, axis=0)
    mi_mean = np.mean(results_mi, axis=0)
    mi_std = np.std(results_mi, axis=0)
    pyplot.errorbar(x=n_states, y=mi_mean, yerr=mi_std, fmt='-')

    pyplot.legend(['naive', 'predicitve'], loc='lower right')

    #
    # 
    #
    #pyplot.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
    pyplot.subplots_adjust(bottom=0.12, top=0.9, wspace=0.1)
    #pyplot.subplots_adjust(wspace=0.1)
    pyplot.show()
    