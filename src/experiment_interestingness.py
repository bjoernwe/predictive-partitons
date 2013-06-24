import matplotlib.pyplot as plt
import numpy as np

import worldmodel

from studienprojekt.env_maze import EnvMaze
from studienprojekt.env_swiss_roll import EnvSwissRoll

if  __name__ =='__main__':

    # sample data
    steps = 4000
    maze = EnvMaze(maximum=10, step_size=0.8, sigma=0.2)
    #maze = EnvSwissRoll(sigma=0.5)
    data, actions = maze.do_random_steps(num_steps=steps)
    
    # model
    model = worldmodel.WorldModelSpectral()
    model.add_data(x=data, actions=actions)
    
    # learn and plot
    P = 4
    for i in range(P):
        model.single_splitting_step()
        
        plt.subplot(2, P, i+1)
        model.plot_tree_data(color='last_gain', show_plot=False)
        #model.plot_state_borders(show_plot=False)
        plt.colorbar()
        
        plt.subplot(2, P, i+P+1)
        model.plot_states(show_plot=False)
        model.plot_state_borders(show_plot=False)
        plt.plot(data[:,0], data[:,1], '.', color='0.0')
    
    plt.show()
