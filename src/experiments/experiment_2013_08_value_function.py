from matplotlib import pyplot

import rl_tools

from studienprojekt import env_maze



if __name__ == '__main__':
    
    maze = env_maze.EnvMaze()
    explorer = rl_tools.RLExploration(model=maze)
    explorer.explore(steps=10000, live_plot=True)
            
    pyplot.ioff()
    pyplot.show()
    