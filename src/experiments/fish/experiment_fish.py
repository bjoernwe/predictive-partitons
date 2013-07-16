from matplotlib import pyplot
import numpy as np
import pickle

import worldmodel


if __name__ == '__main__':

    # load data
    N = 1000
    data = pickle.load(open('experiments/fish/data/views/seq_0000.pckl'))
    data = np.array(data, dtype='float64')
    data = data[0:N,:]
    
    # load ground truth
    # DEFAULT_FEATURE_KEYS = ("id", "x", "y", "z", "phi_x", "phi_y", "phi_z", "scale")
    configs = pickle.load(open('experiments/fish/data/configs/seq_0000.pckl'))
    configs = configs[0:N,0,:]
    
    # initialize model
    model = worldmodel.WorldModelSFA()
    model.add_data(data)
    
    # train world model incrementally
    # 
    # after every learning step we do a regression from the states to the ground
    # truth to evaluate how much (and what) informations the states code. 
    L = 20
    residuals = np.zeros((L, 8))
    for i in range(L):
        model.single_splitting_step(min_gain=0.02)
                
        states = model.classify_to_vector(data)
        regression = np.linalg.lstsq(a=states, b=configs)
        residuals[i] = regression[1] / N
    
    # plot result
    pyplot.plot(residuals)
    pyplot.legend(["id", "x", "y", "z", "phi_x", "phi_y", "phi_z", "scale"])
    pyplot.title('error over time')

    pyplot.figure()
    pyplot.plot(configs)
    pyplot.legend(["id", "x", "y", "z", "phi_x", "phi_y", "phi_z", "scale"])
    pyplot.title('ground truth')
    
    pyplot.figure()
    coeffs = regression[0]
    pyplot.plot(states.dot(coeffs))
    pyplot.title('learned')
    pyplot.show()
    #model.plot_stats(show_plot=True)
    