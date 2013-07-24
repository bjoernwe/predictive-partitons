from matplotlib import pyplot

import studienprojekt.env_maze

import worldmodel


if __name__ == '__main__':
    
    # environment
    maze_size = 20
    env = studienprojekt.env_maze.EnvMaze(maximum=maze_size)
    data, actions = env.do_random_steps(num_steps=2000)
    
    # train model
    model = worldmodel.WorldModelFactorize()
    model.add_data(data=data, actions=actions)
    model.learn(min_gain=0.1)
    
    # plot
    cm = pyplot.cm.get_cmap('summer')
    pyplot.subplot(2, 3, 1)
    pyplot.imshow(-env.Z.T, interpolation='none', origin='lower', extent=[0, maze_size+1, 0, maze_size+1], cmap=cm)
    model.models.values()[0].plot_tree_data(color='none', show_plot=False)
    for i, (a, m) in enumerate(model.models.items()):
        pyplot.subplot(2, 3, i+2+(i/2))
        pyplot.imshow(-env.Z.T, interpolation='none', origin='lower', extent=[0, maze_size+1, 0, maze_size+1], cmap=cm)
        pyplot.title('action: %s' % a)
        m.plot_state_borders(show_plot=False)
    pyplot.show()
    