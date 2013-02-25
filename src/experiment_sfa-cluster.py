import numpy as np

from matplotlib import pyplot

import mdp

from experiment_voronoi import VoronoiData

if __name__ == '__main__':

    # data
    c = 5
    n = 2000
    voronoi = VoronoiData(n=n, k=c, power=5)
    
    # train SFA
    sfa = mdp.nodes.SFANode(output_dim=1)
    sfa.train(voronoi.data)

    # plot result
    delta = 0.025
    x = np.arange(0.1, 1.0, delta)
    y = np.arange(0.1, 1.0, delta)
    xx, yy = np.meshgrid(x, y)
    print xx
    print yy
    z = np.sin(xx**2+yy**2)/(xx**2+yy**2)
    for y in range(0, 1, 0.025):
        for x in range(0, 1, 0.025):
            
     

    # plot training data
    #pyplot.subplot(1,2,1)
    #voronoi.plot()
