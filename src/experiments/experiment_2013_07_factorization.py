from matplotlib import pyplot

import studienprojekt.env_cube

import worldmodel


if __name__ == '__main__':
    
    # environment
    env = studienprojekt.env_cube.EnvCube(step_size=0.1, sigma=0.05)
    data, actions = env.do_random_steps(num_steps=2000)
    
    # train model
    model = worldmodel.WorldModelFactorize()
    model.add_data(data=data, actions=actions)
    
    # learn and plot
    for i in range(3):
        for j, (a, m) in enumerate(model.models.items()):
            m.single_splitting_step()
            mi = worldmodel.WorldModelTree._mutual_information(m.transitions[a])
            pyplot.subplot(3, 4, 4*i + j+1)
            pyplot.title(mi)
            m.plot_states(show_plot=False)
            m.plot_state_borders(show_plot=False)
    pyplot.show()
