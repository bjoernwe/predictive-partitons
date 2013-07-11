import numpy as np

from matplotlib import pyplot
from scipy.sparse import linalg

import worldmodel

import visualization

from studienprojekt.env_cube import EnvCube


if __name__ == '__main__':

    # data
    env = EnvCube(step_size=0.2, sigma=0.05)
    print env.get_available_actions()
    data_all, actions = env.do_random_steps(num_steps=4000)
    
    # graph
    model = worldmodel.WorldModelSpectral()
    model.add_data(x=data_all, actions=actions)
    refs_all, refs_1, P = model._get_transition_graph(action='D0', k=5, fast_partition=True, normalize=False)
    data = model._get_data_for_refs(refs=refs_all)
    
    #print P
    #visualization.plot_graph(means=data, affinity_matrix=P, show_plot=False)
    
    # get eigenvalues
    #E_large, U_large = linalg.eigs(np.array(P), k=2, which='LR')
    #E_small, U_small = linalg.eigs(np.array(P), k=1, which='SR')
    E, U = linalg.eigs(np.array(P), k=15, which='LR')
    U *= np.sqrt(data.shape[0])

    # plot
    pyplot.figure()
    cm = pyplot.cm.get_cmap('summer')
    for j in range(15):
        pyplot.subplot(3, 5, j+1)
        pyplot.scatter(x=data[:,0], y=data[:,1], c=(np.real(U[:,j])), edgecolors='none')#, vmin=-7, vmax=7)
        #pyplot.scatter(x=data[:,0], y=data[:,1], c=np.sign(np.real(U[:,j])), edgecolors='none', cmap=cm)
        pyplot.title(str(E[j]))


    pyplot.colorbar()
    pyplot.show()
    