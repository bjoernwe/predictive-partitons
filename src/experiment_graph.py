import numpy as np
import scipy.spatial

from matplotlib import pyplot
from scipy.sparse import linalg

from studienprojekt import visualization


def get_graph(data, fast_partition, k=5, normalize=False):
    
    # pairwise distances
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)

    # transitions
    N, _ = data.shape
    W = np.zeros((N, N))
    W += 0.001

    # big transition matrix
    # including transitions of the k nearest neighbors
    for i in range(N-1):
        indices = np.argsort(distances[i])  # closest one should be the point itself
        # index: refs -> refs_all
        s = i
        for j in indices[0:k+1]:
            # index: refs -> refs_all
            t = j
            u = i+1
            if fast_partition:
                W[s,u] = -1
                W[u,s] = -1
            else:
                W[s,u] = 1
                W[u,s] = 1
            if s != t:
                W[s,t] = 1
                W[t,s] = 1
            
    # normalize matrix
    if normalize:
        d = np.sum(W, axis=1)
        for i in range(N):
            if d[i] > 0:
                W[i] = W[i] / d[i]
            
    return W
    

if __name__ == '__main__':

    # parameters
    k = 10    
    N = 500
    fast_data = True
    fast_partition = True

    # data    
    mean = np.array([2, 0])
    data = np.zeros((N, 2))
    for i in range(N):
        if fast_data:
            mean *= -1
        else:
            if i == (N//4):
                mean *= -1
        data[i] = np.random.randn(2) + mean
    
    # graph & eigenvectors
    W = get_graph(data, fast_partition=fast_partition, k=k, normalize=True)
    E, U = linalg.eigs(W, k=2, which='LR')
    E = np.real(E)
    U = np.real(U)
    print E

    # plot data and eigenvector
    if fast_partition:
        c = 0
    else:
        c = 1
    pyplot.subplot(1, 3, 1)
    pyplot.scatter(x=data[:,0], y=data[:,1], s=100, c=U[:,c], edgecolor='None')
    pyplot.subplot(1, 3, 2)
    pyplot.scatter(x=data[:,0], y=data[:,1], s=100, c=np.sign(U[:,c]), edgecolor='None')
    
    # plot spectral
    pyplot.subplot(1, 3, 3)
    pyplot.scatter(x=U[:,0], y=U[:,1], s=100)
    
    # plot graph
    #pyplot.figure()
    #visualization.plot_graph(means=data, affinity_matrix=W, show_plot=False)
    
    # show plot
    pyplot.show()
    
    