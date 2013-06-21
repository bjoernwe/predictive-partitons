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
            if s != t:
                W[s,t] = 1
                W[t,s] = 1
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
    k = 10    
    N = 500
    
    for i, fast_data in enumerate([False, True]):
        
        for j, fast_partition in enumerate([False, True]): 

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
            W = get_graph(data, fast_partition=fast_partition, k=k, normalize=True)
            E, U = linalg.eigs(W, k=2, which='LR')
            E, U = np.real(E), np.real(U)
    
            # plot data and eigenvector
            pyplot.figure(0)
            pyplot.subplot(2, 2, 2*i+j+1)
            pyplot.title('data fast: %s / partition fast: %s' % (fast_data, fast_partition))
            if fast_partition:
                pyplot.scatter(x=data[:,0], y=data[:,1], s=100, c=U[:,0], edgecolor='None')
            else:
                pyplot.scatter(x=data[:,0], y=data[:,1], s=100, c=U[:,1], edgecolor='None')
            
            # plot spectral
            pyplot.figure(1)
            pyplot.subplot(2, 2, 2*i+j+1)
            pyplot.title('data fast: %s / partition fast: %s' % (fast_data, fast_partition))
            pyplot.scatter(x=U[:,0], y=U[:,1], s=100)
            
    # show plot
    pyplot.show()
    
    