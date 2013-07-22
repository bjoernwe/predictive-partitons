"""
Factorization of actions

Separate graphs are build for each action and afterwards added together. This
has the drawback that connections are not drawn to the real neighbors but only
to the ones that belong to the same actions. 
"""

import numpy as np
import scipy.linalg
import scipy.spatial

from matplotlib import pyplot

from studienprojekt.env_cube import EnvCube

import worldmodel


def get_graph(model, action, fast_partition, k=5, normalize=False):
    
    [refs_1, refs_2] = model._get_transition_refs_for_action(action=action, heading_in=False, inside=True, heading_out=False)
    #[refs_1, refs_2] = model._get_transition_refs(heading_in=False, inside=True, heading_out=False)
    refs_all = model._get_data_refs()
    data = model._get_data_for_refs(refs_1)
    n_trans = len(refs_1)        
    N = len(refs_all)        
    
    # pairwise distances
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)

    # transition matrix
    W = np.zeros((N, N))
    #W += -1e-8

    # transitions to neighbors
    # s - current node
    # t - neighbor node
    for i in range(n_trans):
        indices = np.argsort(distances[i])  # closest one should be the point itself
        s = refs_all.index(refs_1[i])
        for j in indices[0:k+1]:
            t = refs_all.index(refs_1[j])
            if s != t:
                W[s,t] = 1
                W[t,s] = 1

    # transitions to successors
    # s - current node
    # t - neighbor node
    # u - following node
    for i in range(n_trans):
        indices = np.argsort(distances[i])  # closest one should be the point itself
        s = refs_all.index(refs_1[i])
        for j in indices[0:k+1]:
            #t = refs_all.index(refs_1[j])
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
        for i in range(N):
            if d[i] > 0:
                W[i] = W[i] / d[i]
            
    return W


if __name__ == '__main__':

    # parameters
    steps = 2000
    fast = True
    k = 15
    normalize = True
    smallest = False
    laplacian = False
    plot_sign = True
    
    # data
    #env = EnvCube(step_size=0.2, sigma=0.01)
    env = EnvCube(step_size=0.1, sigma=0.05)
    print env.get_available_actions()
    data, actions = env.do_random_steps(num_steps=steps)
    N, D = data.shape

    # model
    model = worldmodel.WorldModelTree()
    model.add_data(x=data, actions=actions)
    
    # graph
    selected_action = 'D0'
    W = 3 * get_graph(model=model, action=selected_action, fast_partition=True, k=k, normalize=normalize)
    for action in model.get_possible_actions(ignore_none=True):
        if action == selected_action:
            continue
        W += get_graph(model=model, action=action, fast_partition=False, k=k, normalize=normalize)
        
    
    # get eigenvalues
    if laplacian:
        W = np.diag(np.sum(W, axis=1)) - W
    E, U = scipy.linalg.eig(a=W)
    
    # sort eigenvalues
    E = np.real(E)
    idx = np.argsort(np.real(E))
    #idx = np.argsort(np.abs(E))
    
    # plot
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
    