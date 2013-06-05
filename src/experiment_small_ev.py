import numpy as np

from matplotlib import pyplot
from scipy.sparse import linalg

import worldmodel
from studienprojekt.env_cube import EnvCube
from studienprojekt.env_maze import EnvMaze
from studienprojekt.env_noise import EnvNoise
from studienprojekt.env_random_walk import EnvRandomWalk


if __name__ == '__main__':
    
    environments = [EnvCube]#, EnvNoise, EnvRandomWalk, EnvMaze]
    plot_ranges = [[-1, 1], 
                   [0, 1], 
                   [0, 1]]
    plot_action = ['D0', 
               None, 
               0]

    #for i, generator in enumerate(data_generators):
    for i, Environment in enumerate(environments):
        
        # train model
        model = worldmodel.WorldModelSpectral()
        #env = Environment(step_size=0.2, sigma=0.01)
        env = Environment(step_size=0.1, sigma=0.05)
        print env.get_available_actions()
        data, actions = env.do_random_steps(num_steps=4000)
        model.add_data(x=data, actions=actions)
        refs_all, refs_1, P = model._get_transition_graph(action=plot_action[i], k=10, normalize=True)
        data = model._get_data_for_refs(refs=refs_all)
        
        # get eigenvalues
        print P.shape
        #E_large, U_large = linalg.eigs(np.array(P), k=2, which='LR')
        #E_small, U_small = linalg.eigs(np.array(P), k=1, which='SR')
        E, U = linalg.eigs(np.array(P), k=15, which='SR')

        # plot
        pyplot.figure()
        cm = pyplot.cm.get_cmap('summer')
        for j in range(15):
            pyplot.subplot(3, 5, j+1)
            pyplot.scatter(x=data[:,0], y=data[:,1], c=(np.real(U[:,j])), edgecolors='none', vmin=-0.2, vmax=0.2)
            #pyplot.scatter(x=data[:,0], y=data[:,1], c=np.sign(np.real(U[:,j])), edgecolors='none', cmap=cm)
            pyplot.title(str(E[j]))
            print np.min(np.real(U[:,j]))

    pyplot.colorbar()
    pyplot.show()
    