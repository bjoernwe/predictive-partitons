import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation

import environment


class EnvFace(environment.Environment):
    """Returns the video of a face (20x28=560 pixels, 1965 frames).
    """

    def __init__(self, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:        int - 
        """
        super(EnvFace, self).__init__(seed=seed)
        self.video = np.load(os.path.dirname(__file__) + '/faces.npy')
        self.n_frames, self.ndim = self.video.shape
        self.counter = 0
        self.current_state = self.video[self.counter]
        return
    
    
    def _do_action(self, action):
        """Perform the given action and return the resulting state of the
        environment and the reward as well.
        --------------------------------------
        Parameters:
        action:     str - direction of the action to be performed
        --------------------------------------
        Return:
        new_state:    np.ndarray - coordinates of the agent after the step
        reward = 0
        """
        
        self.counter += 1
        if self.counter < self.n_frames:
            self.current_state = self.video[self.counter]
        else:
            print 'Warning: Not more than %d video frames available (%d)!' % (self.n_frames, self.counter) 
            self.current_state = np.zeros(self.ndim)
        return self.current_state, 0



def main():

    nx = 28
    ny = 20
    env = EnvFace()

    fig = plt.figure()
    data = 255 - np.reshape(env.current_state, (nx, ny))
    im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=255)

    def init():
        im.set_data(np.zeros((nx, ny)))
    
    def animate(i):
        data = 255 - np.reshape(env.do_action()[0], (nx, ny))
        im.set_data(data)
        return im
    
    _ = animation.FuncAnimation(fig, animate, init_func=init, frames=nx * ny, interval=25)
    plt.show()
    


if __name__ == '__main__':
    main()
