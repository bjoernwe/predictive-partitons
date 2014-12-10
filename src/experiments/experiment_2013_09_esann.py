"""
Generates data for the ESANN paper.
"""

import cPickle as pickle

from matplotlib import pyplot

import worldmodel
from experiment_2013_02_noisy_dim import NoisyDimData
from experiment_2013_03_random_walk_swissroll import RandomSwissRollData

            
if __name__ == '__main__':
    
    #
    # parameters
    #
    resolution = 1000
    data_size_training = 5000
    data_size_test = 10000
    plot = False

    for i in range(10):
        
        #
        # data
        # 
        data_generators = [RandomSwissRollData,
                           NoisyDimData]
        data_list = []
        data_list_test = []
        for _, generator in enumerate(data_generators):
            data = generator(n=data_size_training, seed=i)
            data_test = generator(n=data_size_test, seed=i+100)
            data_list.append(data)
            data_list_test.append(data_test)
        # scale swiss role
        data_list[0].data += 1.
        data_list[0].data /= 2.
        data_list_test[0].data += 1.
        data_list_test[0].data /= 2.
    
        models = {}
        models['swiss_naive'] = worldmodel.WorldModel(method='naive')
        models['swiss_predictive'] = worldmodel.WorldModel(method='spectral')
        models['noise_naive'] = worldmodel.WorldModel(method='naive')
        models['noise_predictive'] = worldmodel.WorldModel(method='spectral')
    
        #
        # swiss role, naive
        #
        model = models['swiss_naive']
        model.add_data(data=data_list[0].data, actions=data_list[0].actions)
        model.add_test_data(data=data_list_test[0].data)
        model.update_stats()
     
        for _ in range(3):
            for leaf in model.tree.get_leaves():
                leaf._apply_split(allow_useless_split=True)
            for leaf in model.tree.get_leaves():
                leaf._apply_split(allow_useless_split=True)
            model.update_stats()
     
        # plot
        if plot:
            pyplot.subplot(2, 4, 1)
            model.plot_data(color='none', show_plot=False)
            model.plot_state_borders(show_plot=False, resolution=resolution)
            pyplot.title('a')
            pyplot.ylabel('y')
            pyplot.xlabel('x')
             
            pyplot.subplot(2, 2, 3)
            list_mi = [s.mutual_information for s in model.stats]
            list_n = [s.n_states for s in model.stats]
            pyplot.plot(list_n, list_mi, '^-')
            pyplot.legend(['naive', 'predictive'], loc='lower right')
            pyplot.title('e')
            pyplot.xlabel('number of partitions')
            pyplot.ylabel('mututal information')
    
    
        #
        # swiss role, predictive
        #
        model = models['swiss_predictive']
        model.add_data(data=data_list[0].data, actions=data.actions)
        model.add_test_data(data=data_list_test[0].data)
        model.update_stats()
    
        for _ in range(24-1):
            model.single_splitting_step(min_gain=float('-inf'))
            # plot data and borders
            if plot:
                if model.get_number_of_states() == 24:
                    pyplot.subplot(2, 4, 2)
                    model.plot_data(color='none', show_plot=False)
                    #model.plot_states(show_plot=False, resolution=100)
                    model.plot_state_borders(show_plot=False, resolution=resolution)
                    pyplot.title('b')
                    pyplot.xlabel('x')
                    #pyplot.ylabel('y')
                    
        # plot statistics
        if plot:
            pyplot.subplot(2, 2, 3)
            list_mi = [s.mutual_information for s in model.stats]
            list_n = [s.n_states for s in model.stats]
            pyplot.plot(list_n, list_mi, 'o-')
            pyplot.legend(['naive', 'predictive'], loc='lower right')
    
    
        #
        # noise, naive
        #
        model = models['noise_naive']
        model.add_data(data=data_list[1].data, actions=data.actions)
        model.add_test_data(data=data_list_test[1].data)
        model.update_stats()
     
        for _ in range(3):
            for leaf in model.tree.get_leaves():
                leaf._apply_split(allow_useless_split=True)
            for leaf in model.tree.get_leaves():
                leaf._apply_split(allow_useless_split=True)
            model.update_stats()
     
        # plot
        if plot:
            pyplot.subplot(2, 4, 3)
            #model.plot_data(color='silver', show_plot=False)
            model.plot_state_borders(show_plot=False, resolution=resolution)
            pyplot.title('c')
            pyplot.xlabel('x')
            #pyplot.ylabel('y')
             
            pyplot.subplot(2, 2, 4)
            list_mi = [s.mutual_information for s in model.stats]
            list_n = [s.n_states for s in model.stats]
            pyplot.plot(list_n, list_mi, '^-')
            pyplot.legend(['naive', 'predictive'], loc='lower right')
            pyplot.title('f')
            pyplot.xlabel('number of partitions')
            pyplot.ylabel('mututal information')
     
     
        #
        # noise, predictive
        #
        model = models['noise_predictive']
        model.add_data(data=data_list[1].data, actions=data.actions)
        model.add_test_data(data=data_list_test[1].data)
        model.update_stats()
     
        for _ in range(16-1):
            model.single_splitting_step(min_gain=float('-inf'))
            # plot data and borders
            if plot:
                if model.get_number_of_states() == 8:
                    pyplot.subplot(2, 4, 4)
                    #model.plot_data(color='none', show_plot=False)
                    model.plot_state_borders(show_plot=False, resolution=resolution)
                    pyplot.title('d')
                    pyplot.xlabel('x')
                    #pyplot.ylabel('y')
                     
        # plot statistics
        if plot:
            pyplot.subplot(2, 2, 4)
            list_mi = [s.mutual_information for s in model.stats]
            list_n = [s.n_states for s in model.stats]
            pyplot.plot(list_n, list_mi, 'o-')
            pyplot.legend(['naive', 'predictive'], loc='lower right')
         
            #pyplot.tight_layout()
            pyplot.subplots_adjust(wspace=0.2, hspace=0.3)
            pyplot.show()
    
        # save results
        pickle.dump(models, open('experiment_2014_04_08_esann_models_%d.dump' % i, 'wb') )
        pickle.dump(data_list, open('experiment_2014_04_08_esann_data_%d.dump' % i, 'wb') )
        