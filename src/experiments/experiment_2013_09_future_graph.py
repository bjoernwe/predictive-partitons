"""
We construct a graph that is intended to bind together points that have a
similar future. We see from the plots that this way no distinction is necessary
between fast and slow changing data.
"""

import numpy as np
import scipy.spatial

from matplotlib import pyplot
from scipy.sparse import linalg


def get_graph(data, k=5, normalize=False):
    
    # pairwise distances
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)

    # transitions
    N, _ = data.shape
    W = np.zeros((N, N))
    W += 0.001

    # including transitions of the k nearest neighbors
    # i - current state
    # j - neighbor state
    for i in range(N-1):
        indices = np.argsort(distances[i])  # closest one should be the point itself
        for j in indices[0:k+1]:
            if i != j:
                W[i,j] = 1
                W[j,i] = 1
            
    # including transitions to points with similar future
    # i - current state
    # j - successor state
    # t - neighbor of j
    # s - predecessors of t (points with similar future as i)
    for i in range(N-1):
        j = i+1
        indices = np.argsort(distances[j])  # closest one should be the point itself
        for t in indices[0:k+1]:
            s = t-1
            if s != t:
                W[i,s] = 1
                W[s,i] = 1
            
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
    N = 2000
    
    for i, fast_data in enumerate([False, True]):
        
        # data    
        mean = np.array([2, 0])
        data = np.zeros((N, 2))
        for l in range(N):
            if fast_data:
                mean *= -1
            else:
                if l == (N//4):
                    mean *= -1
            data[l] = np.random.randn(2) + mean

        # graph & eigenvectors
        W = get_graph(data, k=k, normalize=True)
        E, U = linalg.eigs(W, k=2, which='LR')
        E, U = np.real(E), np.real(U)
        print E

        # plot data and eigenvector
        pyplot.figure(0)
        pyplot.subplot(1, 2, i+1)
        pyplot.title('data fast: %s' % (fast_data))
        pyplot.scatter(x=data[:,0], y=data[:,1], s=100, c=U[:,1], edgecolor='None')
        
    # show plot
    pyplot.show()
    
    