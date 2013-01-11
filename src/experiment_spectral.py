import numpy as np

from matplotlib import pyplot
from scipy.sparse import linalg

from experiment_voronoi import VoronoiData


if __name__ == '__main__':

    c = 8
    n = 5000
    voronoi = VoronoiData(n=n, k=c, power=5)
    
    # transitions
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i+1)%n] = 1
    
    # add neighborhood affinity to transitions matrix
    for i in range(n):
        distances = np.sqrt(((np.array(voronoi.data) - voronoi.data[i])**2).sum(axis=1))
        indices = np.argsort(distances)
        neighbors = indices[1:10+1]
        for j in neighbors:
            W[i,j] = 1.
            W[j,i] = 1.
            
    # normalize matrix
    d = np.sum(W, axis=1)
    D = np.diag(d)
    L = D - W
    Lrw = L / d[:,np.newaxis]

    # eigenvector
    E, U = linalg.eigen(Lrw, k=2, which='SM')
    idx = np.argsort(E)

    # plot result
    pyplot.subplot(1,2,2)
    for i in range(n):
        x = voronoi.data[i,0]
        y = voronoi.data[i,1]
        if U[:,idx[1]][i].real >= 0:
            pyplot.plot(x, y, 'bo')
        else: 
            pyplot.plot(x, y, 'rs')

    pyplot.subplot(1,2,1)
    voronoi.plot()
    #pyplot.show()
    
