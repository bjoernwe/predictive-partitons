"""
Experiments for the extended abstract for the ICML workshop on spectral 
learning.
"""

from matplotlib import pyplot

import worldmodel
import experiment_2013_02_noisy_dim
import experiment_2013_03_random_walk_2d
import experiment_2013_03_random_walk_swissroll

            
if __name__ == '__main__':

    data_generators = [experiment_2013_03_random_walk_swissroll.RandomSwissRollData,
                       experiment_2013_02_noisy_dim.NoisyDimData,
                       experiment_2013_03_random_walk_2d.RandomWalk2DData]
    plot_ranges = [[-1, 1], [0, 1], [0, 1]]
    data_sizes = [4000, 4000, 4000]
    titles = ['(a)', '(b)', '(c)']

    for i, generator in enumerate(data_generators):
        
        # train model
        data = generator(n=data_sizes[i])
        model = worldmodel.WorldModel(method='spectral')
        model.add_data(data=data.data, actions=data.actions)
        model.learn(min_gain=0.02)
        
        # plot data and result
        pyplot.subplot(1, 3, i+1)
        model.plot_state_borders(show_plot=False)
        if i == 0:
            model.plot_data(color='none', show_plot=False)
        pyplot.xlabel(titles[i])
        
    pyplot.show()
    