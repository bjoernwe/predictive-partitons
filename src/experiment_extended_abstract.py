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
    titles = ['(a)', '(b)', '(c)']

    for i, generator in enumerate(data_generators):
        
        # train model
        data = generator(n=data_sizes[i])
        model = worldmodel.WorldModelTree()
        model.add_data(x=data.data, actions=data.actions)
        model.learn(min_gain=0.02, max_costs=.02)
        
        # plot data and result
        pyplot.subplot(1, 3, i+1)
        model.plot_states(show_plot=False, range_x=plot_ranges[i], range_y=plot_ranges[i])
        if i == 0:
            model.plot_tree_data(color_coded=False, show_plot=False)
        pyplot.xlabel(titles[i])
        
    pyplot.show()
    