"""
Factorization of actions

First all the neighbors are connected. Then fore each action the corresponding
connections are set.
"""

import numpy as np
import scipy.linalg
import scipy.spatial

from matplotlib import pyplot

from studienprojekt.env_cube import EnvCube

import worldmodel


def get_graph(model, fast_action, k=15, normalize=True):

    # complete data and action sets    
    refs_all = model._get_data_refs()
    data_all = model._get_data_for_refs(refs_all)
    actions = model.root().actions
    assert data_all.shape[0] == len(actions)
    
    # pairwise distances
    distances_all = scipy.spatial.distance.pdist(data_all)
    distances_all = scipy.spatial.distance.squareform(distances_all)

    # action-specific data
    data = {}
    refs_1 = {}
    distances = {}
    possible_actions = model.get_possible_actions(ignore_none=True)
    for action in possible_actions:
        r1, _ = model._get_transition_refs_for_action(action=action, heading_in=False, inside=True, heading_out=True)
        refs_1[action] = r1
        data[action] = model._get_data_for_refs(refs_1[action])
        dist = scipy.spatial.distance.pdist(data[action])
        distances[action] = scipy.spatial.distance.squareform(dist)
            
    # transition matrix
    N, _ = data_all.shape
    W = np.zeros((N, N))
    #W += -1e-8
    
    # transitions to neighbors
    # s - current node
    # t - neighbor node
    n_possible_actions = len(possible_actions)
    l = k * n_possible_actions
    for s in range(N-1):
        indices = np.argsort(distances_all[s])  # closest one should be the point itself
        for t in indices[0:l+1]:
            if s != t:
                W[s,t] = 1
                W[t,s] = 1

    # transitions to successors
    # s - current node
    # t - neighbor node
    # u - following node
    for a in possible_actions:
        for i, r1 in enumerate(refs_1[a]):
            indices = np.argsort(distances[a][i])  # closest one should be the point itself
            s = refs_all.index(r1)
            for j in indices[0:k+1]:
                t = refs_all.index(refs_1[a][j])
                u = t+1
                if a == fast_action:
                    W[s,u] = -(n_possible_actions - 1)
                    W[u,s] = -(n_possible_actions - 1)
                else:
                    W[s,u] = 1
                    W[u,s] = 1
                
    # normalize matrix
    if normalize:
        d = np.sum(W, axis=1)
        for i in range(N):
            if d[i] > 0:
                W[i] = W[i] / d[i]
            
    return W


if __name__ == '__main__':

    # 
    # parameters for experiment
    # 
    steps = 2000
    fast = True
    k = 10
    normalize = True
    smallest = False
    laplacian = False
    plot_sign = True
    
    #
    # generate data
    # 
    #env = EnvCube(step_size=0.2, sigma=0.01)    # simple
    env = EnvCube(step_size=0.1, sigma=0.05)    # noisy
    print env.get_available_actions()
    data, actions = env.do_random_steps(num_steps=steps)
    N, D = data.shape

    # 
    # model
    # 
    model = worldmodel.WorldModelTree()
    model.add_data(x=data, actions=actions)
    
    #
    # get eigenvalues of graph (laplacian)
    #
    W = get_graph(model=model, fast_action='D0', k=k, normalize=normalize)
    if laplacian:
        W = np.diag(np.sum(W, axis=1)) - W
    E, U = scipy.linalg.eig(a=W)
    
    #
    # sort the eigenvalues
    #
    E = np.real(E)
    idx = np.argsort(np.real(E))

    #    
    # plot result
    #
    print 'steps=%d, k=%d, fast=%s,\n normalize=%s, smallest=%s' % (steps, k, fast, normalize, smallest)
    pyplot.figure()
    cm = pyplot.cm.get_cmap('summer')
    for i in range(15):
        if smallest:
            j = idx[i]      # smallest
        else:
            j = idx[-i-1]   # largest
        pyplot.subplot(3, 5, i+1)
        if plot_sign:
            pyplot.scatter(x=data[:,0], y=data[:,1], c=np.sign(np.real(U[:,j])), edgecolors='none', cmap=cm)
        else:
            pyplot.scatter(x=data[:,0], y=data[:,1], c=(np.real(U[:,j])), edgecolors='none')#, vmin=-7, vmax=7)
        pyplot.title(str(E[j]))
        pyplot.colorbar()
        
    pyplot.show()
    