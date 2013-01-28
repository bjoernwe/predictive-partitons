import numpy as np

from matplotlib import pyplot
from scipy import sparse

from experiment_voronoi import VoronoiData


if __name__ == '__main__':

    c = 5
    n = 1000
    voronoi = VoronoiData(n=n, k=c, power=5)
    
    # transitions
    W = np.zeros((n, n))
    #for i in range(n):
        #W[i, (i+1)%n] = 1
        #W[(i+1)%n, i] = 1
    
    # add neighborhood affinity to transitions matrix
    k = 10
    for i in range(n):
        distances = np.sqrt(((np.array(voronoi.data) - voronoi.data[i])**2).sum(axis=1))
        indices = np.argsort(distances)
        neighbors = indices[0:k+1]
        print i, neighbors
        # add transitions to neighbors
        #for j in neighbors[1:]:
        #    W[i,j] = .1
        #    W[j,i] = .1
        # add transitions of neighbors to current point
        for j in neighbors[0:]:
            W[i,(j+1)%n] = .01
        #    W[(j+1)%n,i] = .1
            W[j, (i+1)%n] = .01
            
    # Jacobian
    d = np.sum(W, axis=1)
    D = np.diag(d)
    L = D - W
    #L = W
    Lrw = L / d[:,np.newaxis]
    
    # 
    P = W / d[:,np.newaxis]
    Q = P.conj().T.dot(P) - P - P.conj().T + np.eye(n)
    
    if True:
        # eigenvector
        E, U = sparse.linalg.eigen(P, k=3, which='LM')
        #E, U = np.linalg.eig(P)
        idx = np.argsort(abs(E))
        u1 = U[:,idx[-2]] # gross
        u2 = U[:,idx[-3]] # gross
    else:
        # eigenvector
        E, U = sparse.linalg.eigen(Lrw, k=3, which='SM')
        idx = np.argsort(abs(E))
        u1 = U[:,idx[1]] # gross
        u2 = U[:,idx[2]] # gross

    # plot result
    pyplot.subplot(1,3,2)
    for i in range(n):
        x = voronoi.data[i,0]
        y = voronoi.data[i,1]
        if u1[i].real >= 0:
            pyplot.plot(x, y, 'bo')
        else: 
            pyplot.plot(x, y, 'rs')
            
    print voronoi.data.shape
    print U.shape
    pyplot.subplot(1,3,3)
    #pyplot.plot(U[:,idx[1]], U[:,idx[2]], 'o')
    pyplot.plot(u1, u2, 'o')

    pyplot.subplot(1,3,1)
    voronoi.plot()
    pyplot.show()
    
    
    
