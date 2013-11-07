from studienprojekt import env_mario

import rl_tools

if __name__ == '__main__':
    
    mario = env_mario.EnvMario()
    explorer = rl_tools.RLExploration(model=mario)
    explorer.explore(steps=10000, live_plot=True)