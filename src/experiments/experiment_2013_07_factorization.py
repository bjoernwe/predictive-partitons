import numpy as np
import scipy.linalg

from matplotlib import pyplot

import studienprojekt.env_cube

import worldmodel


if __name__ == '__main__':
    
    # environment
    #env = studienprojekt.env_cube.EnvCube(step_size=0.2, sigma=0.01)
    env = studienprojekt.env_cube.EnvCube(step_size=0.1, sigma=0.05)
    data, actions = env.do_random_steps(num_steps=2000)
    
    # train model
    model = worldmodel.WorldModelFactorize()
    model.add_data(data=data, actions=actions)
    model.learn()
    
    for i, (a, m) in enumerate(model.models.items()):
        mi = worldmodel.WorldModelTree._mutual_information(m.transitions[a])
        pyplot.subplot(2, 2, (i+1))
        pyplot.title(mi)
        m.plot_states(show_plot=False)
        #m.plot_tree_data(color='none', show_plot=False)
    pyplot.show()
    