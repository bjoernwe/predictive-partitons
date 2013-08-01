from matplotlib import pyplot

import studienprojekt.env_cube

import worldmodel


if __name__ == '__main__':
    
    # environment
    #env = studienprojekt.env_cube.EnvCube(step_size=0.2, sigma=0.01)
    env = studienprojekt.env_cube.EnvCube(step_size=0.1, sigma=0.05)
    data, actions = env.do_random_steps(num_steps=2000)
    
    # train model
    model = worldmodel.WorldModelFactorize()
    model.add_data(data=data, actions=actions)
    
    # learn & plot
    for i in range(4):
        model.single_splitting_step(action=None, min_gain=0.0)
        for j, (a, m) in enumerate(model.models.items()):
            mi = worldmodel.WorldModelTree._mutual_information(m.transitions[a])
            pyplot.subplot(4, 4, (4*i)+(j+1))
            pyplot.title(mi)
            m.plot_states(show_plot=False)
            m.plot_state_borders(show_plot=False)
    pyplot.show()
    