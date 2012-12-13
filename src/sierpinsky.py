from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import random

def sierpinsky_square(N, iterations=200000):
    """
    Generates a Sierpinski square.
    http://eldar.mathstat.uoguelph.ca/dashlock/Outreach/Articles/GSF.html
    """
    
    A = np.zeros((N,N))
    corners = [np.array([0, 0]),
               np.array([0, N]),
               np.array([N, 0]),
               np.array([N, N])]
    colors = [0., 1., 1., 0.]
    current_position = np.array([0, 0])
    current_color = 0
    
    for _ in range(iterations):
        index = random.randint(0, 3) 
        current_corner = corners[index]
        current_position = (current_position + current_corner) // 2
        current_color = (current_color + colors[index]) / 2.
        x = int(current_position[0])
        y = int(current_position[1])
        A[x, y] = current_color
        
    # normalize
    #weights = np.sum(A, axis=1)
    #A = A / weights[:,np.newaxis]
    return A
    

if __name__ == '__main__':
    
    A = sierpinsky_square(N=16)    
    plt.imshow(A)
    plt.show()
    