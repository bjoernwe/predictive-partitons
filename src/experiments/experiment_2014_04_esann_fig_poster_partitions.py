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
    resolution = 1000


    #
    # load data
    #
    results = {}
    for t in range(T):
        results[t] = {}
        results[t]['models'] = pickle.load(open('experiment_2014_04_esann_models_%d.dump' % t, 'rb') )
        print len(results[t]['models']['swiss_predictive'].stats)
        print len(results[t]['models']['noise_predictive'].stats)
        print ''


    pyplot.figure()
    results[0]['models']['swiss_predictive'].plot_states(show_plot=False, resolution=resolution)

    #pyplot.figure()
    #results[0]['models']['swiss_naive'].plot_states(show_plot=False)

    pyplot.figure()
    results[0]['models']['noise_predictive'].plot_states(show_plot=False, resolution=resolution)

    #pyplot.figure()
    #results[0]['models']['noise_naive'].plot_states(show_plot=False)
    

    #
    # 
    #
    #pyplot.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
    pyplot.subplots_adjust(bottom=0.12, top=0.9, wspace=0.1)
    #pyplot.subplots_adjust(wspace=0.1)
    pyplot.show()
    