from matplotlib import pyplot as plt

import worldmodel

from studienprojekt.env_cube import EnvCube


if __name__ == '__main__':

    env = EnvCube(step_size=0.05, sigma=.002)
    available_actions = env.get_available_actions()
    data, actions = env.do_random_steps(num_steps=4000)
    
    for i, a in enumerate(available_actions):
        plt.subplot(2, 2, i+1)
        plt.title(a)
        model = worldmodel.WorldModelSpectral()
        model.add_data(x=data, actions=actions)
        for _ in range(5):
            model.single_splitting_step(action=a)
        model.plot_states(show_plot=False)
        
    plt.show()
    