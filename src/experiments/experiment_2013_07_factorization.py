"""
Visualizes the largest/smallest eigenvectors of a graph.
"""

import numpy as np
import scipy.linalg
import scipy.spatial

from matplotlib import pyplot

from studienprojekt.env_cube import EnvCube

import worldmodel


def get_graph(model, action, fast_partition, k=5, normalize=False):

    #
    #[refs_1, refs_2] = model._get_transition_refs_for_action(action=action, heading_in=False, inside=True, heading_out=False)
    [refs_1, refs_2] = model._get_transition_refs(heading_in=False, inside=True, heading_out=False)
    data = model._get_data_for_refs(refs_1)
    refs_all = model._get_data_refs()
    n_trans_all = len(refs_all)
    n_trans = len(refs_1)
    
    # pairwise distances
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    
    # transitions
    W = np.zeros((n_trans_all, n_trans_all))
    W += 1e-8
    
    # big transition matrix
    # adding transitions to the k nearest neighbors
    for i in range(n_trans):
        indices = np.argsort(distances[i])  # closest one should be the point itself
        # index: refs -> refs_all
        s = refs_all.index(refs_1[i])
        for j in indices[0:k+1]:
            # index: refs -> refs_all
            t = refs_all.index(refs_1[j])
            if s != t:
                W[s,t] = 1
                W[t,s] = 1
                
    # big transition matrix
    # adding transitions of the k nearest neighbors
    for i in range(n_trans):
        indices = np.argsort(distances[i])  # closest one should be the point itself
        # index: refs -> refs_all
        s = refs_all.index(refs_1[i])
        for j in indices[0:k+1]:
            # index: refs -> refs_all
            u = refs_all.index(refs_2[j])
            if fast_partition:
                W[s,u] = -1
                W[u,s] = -1
            else:
                W[s,u] = 1
                W[u,s] = 1
                
    # normalize matrix
    if normalize:
        d = np.sum(W, axis=1)
        for i in range(n_trans_all):
            if d[i] > 0:
                W[i] = W[i] / d[i]
            
    return W


if __name__ == '__main__':

    # parameters
    steps = 3000
    k = 15

    # data
    env = EnvCube(step_size=0.2, sigma=0.01)
    #env = EnvCube(step_size=0.1, sigma=0.05)
    available_actions = env.get_available_actions()
    data, actions = env.do_random_steps(num_steps=steps)
    
    # train model
    model = worldmodel.WorldModelTree()
    model.add_data(x=data, actions=actions)
    
    # get graph
    selected_action = 'D0'
    W = 1 * get_graph(model=model, action=selected_action, fast_partition=True, k=k, normalize=True)
    #for action in available_actions:
    #    if action == selected_action:# or action == 'U0':
    #        continue
    #    W += get_graph(model=model, action=action, fast_partition=False, k=k, normalize=True)
        
    # eigenvectors
    E, U = scipy.linalg.eig(a=W)
    E = np.real(E)
    idx = np.argsort(np.abs(E))
    
    # plot
    pyplot.figure()
    cm = pyplot.cm.get_cmap('summer')
    for i in range(15):
        #j = idx[i]    # smallest
        j = idx[-i-1]   # largest
        pyplot.subplot(3, 5, i+1)
        #pyplot.scatter(x=data[:,0], y=data[:,1], c=(np.real(U[:,j])), edgecolors='none')#, vmin=-7, vmax=7)
        pyplot.scatter(x=data[:,0], y=data[:,1], c=np.sign(np.real(U[:,j])), edgecolors='none', cmap=cm)
        pyplot.title(str(E[j]))

    pyplot.colorbar()
    pyplot.show()
    