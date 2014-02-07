from matplotlib import pyplot

import worldmodel

from envs import env_circle
from envs import env_cube
from envs import env_noise


if __name__ == '__main__':
    
    env = env_circle.EnvCircle(seed=None)
    #env = env_cube.EnvCube(step_size=0.1, sigma=0.1, ndim=2, seed=None)
    #env = env_noise.EnvNoise(sigma=0.1, ndim=2, seed=None)
    data, actions, _ = env.do_random_steps(num_steps=10000)
    
    print data
    print map(lambda a: env.get_actions_dict()[a], actions)

    model = worldmodel.Worldmodel(method='fast', uncertainty_prior=0, seed=None)
    model.add_data(data=data, actions=actions)
    
    for i in range(3):
        model.split(action=None, min_gain=float('-inf'))
    
    for i in range(env.get_number_of_possible_actions()):
        print env.get_actions_dict()[i], ':', model.partitionings[i].tree._split_params._test_params    
        pyplot.subplot(2, 2, i+1)
        pyplot.title(env.get_actions_dict()[i])
        model.plot_data_colored_for_state(active_action=i, show_plot=False)
    
    pyplot.show()
    