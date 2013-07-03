"""
Generates the plots shown at my INI talk.
"""

from matplotlib import pyplot

import worldmodel
import experiment_noisy_dim
import experiment_random_walk_2d
import experiment_random_walk_swissroll

            
if __name__ == '__main__':

    data_generators = [experiment_random_walk_swissroll.RandomSwissRollData,
                       experiment_noisy_dim.NoisyDimData,
                       experiment_random_walk_2d.RandomWalk2DData]
    plot_ranges = [[-1, 1], [0, 1], [0, 1]]
    data_sizes = [4000, 4000, 4000]

    for i, generator in enumerate(data_generators):
        
        # train model
        data = generator(n=data_sizes[i], seed=1)
        model = worldmodel.WorldModelSpectral()
        model.add_data(x=data.data, actions=data.actions)
        model.learn(min_gain=0.02)
        
        # plot data and result
        pyplot.figure()
        model.plot_state_borders(show_plot=False, resolution=100)
        pyplot.xlabel('feature 1')
        pyplot.ylabel('feature 2')
        #if i == 0:
        pyplot.figure()
        model.plot_tree_data(color='none', show_plot=False)
        pyplot.xlabel('feature 1')
        pyplot.ylabel('feature 2')
        
        # plot mutual information
        pyplot.figure()
        pyplot.plot([s.mutual_information for s in model.stats])
        pyplot.xlabel('learning steps')
        pyplot.ylabel('mutual information')
        
    pyplot.show()
    