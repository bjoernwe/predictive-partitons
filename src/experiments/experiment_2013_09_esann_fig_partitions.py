"""
Generates plots for the ESANN paper.
"""

from matplotlib import pyplot

import worldmodel
from experiment_2013_02_noisy_dim import NoisyDimData
from experiment_2013_03_random_walk_swissroll import RandomSwissRollData

            
if __name__ == '__main__':
    
    #
    # parameters
    #
    resolution = 1000
    data_all = 5000
    data_training = 5000

    #
    # data
    # 
    data_generators = [RandomSwissRollData,
                       NoisyDimData]
    data_list = []
    for i, generator in enumerate(data_generators):
        data = generator(n=data_all, seed=0)
        data_list.append(data)
    # scale swiss role
    data_list[0].data += 1.
    data_list[0].data /= 2.

    models = {}
    models['swiss_naive'] = worldmodel.WorldModel(method='naive')
    models['swiss_predictive'] = worldmodel.WorldModel(method='spectral')
    models['noise_naive'] = worldmodel.WorldModel(method='naive')
    models['noise_predictive'] = worldmodel.WorldModel(method='spectral')


    #
    # swiss role, predictive
    #
    model = models['swiss_predictive']
    model.add_data(data=data_list[0].data[:data_training], actions=data.actions)
    model.update_stats()

    for i in range(16-1):
        model.single_splitting_step(min_gain=float('-inf'))
        
    #pyplot.subplot(1, 3, 1)
    pyplot.figure()
    model.plot_data(color='none', show_plot=False)
    model.plot_states(show_plot=False, resolution=resolution, range_x=[0,1], range_y=[0,1])
    model.plot_state_borders(show_plot=False, resolution=resolution, range_x=[0,1], range_y=[0,1])
    #pyplot.title('A')
    pyplot.xlabel('x')
    pyplot.ylabel('y')

                
    #
    # noise, predictive
    #
    model = models['noise_predictive']
    model.add_data(data=data_list[1].data[:data_training], actions=data.actions)
    model.update_stats()
  
    for _ in range(8-1):
        model.single_splitting_step(min_gain=float('-inf'))
        # plot data and borders
 
    #pyplot.subplot(1, 3, 2)
    pyplot.figure()
    model.plot_states(show_plot=False, resolution=resolution, range_x=[0,1], range_y=[0,1])
    model.plot_state_borders(show_plot=False, resolution=resolution, range_x=[0,1], range_y=[0,1])
    #pyplot.title('B')
    pyplot.xlabel('x')
    pyplot.ylabel('y (noise)')
                  
    #
    # noise, naive
    #
    model = models['noise_naive']
    model.add_data(data=data_list[1].data, actions=data.actions)
    model.update_stats()
  
    for _ in range(3):
        for leaf in model.tree.get_leaves():
            leaf._apply_split(allow_useless_split=True)
        for leaf in model.tree.get_leaves():
            leaf._apply_split(allow_useless_split=True)
        model.update_stats()
  
    # plot
    #pyplot.subplot(1, 3, 3)
    #pyplot.figure()
    #model.plot_data(color='silver', show_plot=False)
    #model.plot_states(show_plot=False, resolution=resolution)
    #pyplot.title('A')
    #pyplot.xlabel('x')
    #pyplot.ylabel('y')
     
 
    #pyplot.tight_layout()
    #pyplot.subplots_adjust(wspace=0.2, hspace=0.3)
    #pyplot.subplots_adjust(bottom=0.12, top=0.9, wspace=0.1)
    pyplot.show()
