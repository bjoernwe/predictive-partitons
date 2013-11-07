"""
Generates the plots shown at my INI talk.
"""

from matplotlib import pyplot

import worldmodel
from experiment_2013_02_noisy_dim import NoisyDimData
from experiment_2013_03_random_walk_swissroll import RandomSwissRollData

            
if __name__ == '__main__':

    data_generators = [RandomSwissRollData,
                       NoisyDimData]
    #plot_ranges = [[0, 1], [0, 1]]
    #data_sizes = [4000, 4000]

    data_list = []
    for i, generator in enumerate(data_generators):
        data = generator(n=10000, seed=1)
        data_list.append(data)
        
    # 
    data_list[0].data += 1.
    data_list[0].data /= 2.
    
    for i, data in enumerate(data_list):
        
        pyplot.figure()
            
        for j, method in enumerate(['naive', 'spectral']):
        
            model = worldmodel.WorldModel(method=method)
            model.add_data(data=data.data, actions=data.actions)
            model.update_stats()
            
            if method == 'naive':
                
                for _ in range(3):
                    for leaf in model.tree.get_leaves():
                        leaf._apply_split(allow_useless_split=True)
                    for leaf in model.tree.get_leaves():
                        leaf._apply_split(allow_useless_split=True)
                    model.update_stats()

                # plot
                pyplot.subplot(2, 2, j+1)
                model.plot_data(color='silver', show_plot=False)
                model.plot_state_borders(show_plot=False, resolution=100)
                pyplot.subplot(2, 1, 2)
                list_mi = [s.mutual_information for s in model.stats]
                list_n = [s.n_states for s in model.stats]
                pyplot.plot(list_n, list_mi, '^-')
                    
            else:
                
                for k in range(4**2-1):
                    model.single_splitting_step(min_gain=0.0)
                    # plot data and borders
                    if k == 7:
                        pyplot.subplot(2, 2, j+1)
                        model.plot_data(color='none', show_plot=False)
                        model.plot_state_borders(show_plot=False, resolution=100)
                        
                # plot statistics
                pyplot.subplot(2, 1, 2)
                list_mi = [s.mutual_information for s in model.stats]
                list_n = [s.n_states for s in model.stats]
                pyplot.plot(list_n, list_mi, 'o-')
                         
            # plot
            pyplot.legend(['naive', 'predictive'], loc='lower right')
            #pyplot.subplot(2, 1, 2)
            #list_mi = [s.mutual_information for s in model.stats]
            #list_n = [s.n_states for s in model.stats]
            #pyplot.plot(list_n, list_mi, '^-')
            
    pyplot.show()
        

#     for i, generator in enumerate(data_generators):
#         
#         # train model
#         data = generator(n=data_sizes[i], seed=1)
#         if i == 0:
#             data.data += 1.
#             data.data /= 2.
#         
#         model = worldmodel.WorldModel(method='naive')
#         model.add_data(data=data.data, actions=data.actions)
#         model.update_stats()
#         #model.learn(min_gain=0.02)
#         for _ in range(6):
#             for leaf in model.tree.get_leaves():
#                 leaf._apply_split(allow_useless_split=True)
#             model.update_stats()
#         
#         # plot data and result
#         pyplot.figure()
#         model.plot_state_borders(show_plot=False, resolution=100)
#         pyplot.xlabel('feature 1')
#         pyplot.ylabel('feature 2')
#         #if i == 0:
#         #pyplot.figure()
#         model.plot_data(color='none', show_plot=False)
#         #pyplot.xlabel('feature 1')
#         #pyplot.ylabel('feature 2')
#         
#         # plot mutual information
#         list_mi = [s.mutual_information for s in model.stats]
#         list_n = [s.n_nodes for s in model.stats]
#         pyplot.figure()
#         pyplot.plot(list_n, list_mi)
#         pyplot.xlabel('learning steps')
#         pyplot.ylabel('mutual information')
#         
#     pyplot.show()
#     