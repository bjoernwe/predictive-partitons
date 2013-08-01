"""
Factorization of actions

Builds the graph by iterating through all nodes and setting to connections
according to the corresponding action. This is the simplest version.
"""

import numpy as np
import scipy.linalg
import scipy.spatial

from matplotlib import pyplot

from studienprojekt.env_cube import EnvCube

import worldmodel


def get_graph(model, fast_action, k=15, normalize=True):

    refs = model._get_data_refs()
    data = model._get_data_for_refs(refs)
    actions = model._get_actions_for_refs(refs)
    N = len(refs)        
    
    # pairwise distances
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)

    # transition matrix
    W = np.zeros((N, N))
    #W += -1e-8
    
    # number of actions
    action_set = set(actions)
    if None in action_set:
        action_set.remove(None)
    n_actions = len(action_set)
    weight = n_actions - 1

    # transitions to neighbors
    # s - current node
    # t - neighbor node
    # u - following node
    for s in range(N):
        indices = np.argsort(distances[s])  # closest one should be the point itself
        for t in indices[0:k+1]:
            if s != t:
                W[s,t] = 1
                W[t,s] = 1

    # transitions to successors
    # s - current node
    # t - neighbor node
    # u - following node
    for s in range(N-1):
        indices = np.argsort(distances[s])  # closest one should be the point itself
        for t in indices[0:k+1]:
            u = t+1
            if u >= N:
                continue
            if actions[u] == fast_action:
                W[s,u] = -weight
                W[u,s] = -weight
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
    
    # model
    model = worldmodel.WorldModelTree()
    model.add_data(data, actions=actions)
    
    # get eigenvalues
    W = get_graph(model=model, fast_action='D0', k=k, normalize=normalize)
    if laplacian:
        W = np.diag(np.sum(W, axis=1)) - W
    E, U = scipy.linalg.eig(a=W)
    
    # sort eigenvalues
    E = np.real(E)
    idx = np.argsort(np.real(E))
    #idx = np.argsort(np.abs(E))
    
    # plot
    print 'steps=%d, k=%d,\n normalize=%s, smallest=%s' % (steps, k, normalize, smallest)
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
    