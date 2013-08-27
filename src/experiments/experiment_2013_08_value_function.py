import numpy as np

from matplotlib import pyplot

import worldmodel

from studienprojekt import env_maze
from studienprojekt import env_model


class PriorityQueue(object):
    """
    Implements a queue from which always the entry with highest probability is
    returned via pop().
    """
    
    def __init__(self):
        self.priorities = []
        self.values = []
        
    def add(self, p, s):
        insert_index = np.searchsorted(self.priorities, p)
        self.priorities.insert(insert_index, p)
        self.values.insert(insert_index, s)
        return
    
    def pop(self):
        p = self.priorities.pop()
        s = self.values.pop()
        return p, s
    
    def len(self):
        return len(self.priorities)
    
    def is_empty(self):
        if len(self.priorities) > 0:
            return False
        else:
            return True
        
        
def QtoV(Q):
    """
    Calculates a value function from Q (assuming greed strategy).
    """
    N = Q.itervalues().next().shape[0]
    V = np.zeros(N)
    for a in Q.keys():
        V = np.maximum(V, Q[a])
    return V
        
        
def plot_value_function(world_model, Q, action=None):
    """
    Plots data of worldmodel colored with value function from Q.
    """
    if action is None:
        V = QtoV(Q)
    else:
        V = Q[action]
    colormap = pyplot.cm.get_cmap('summer')
    for i, leaf in enumerate(world_model.tree.get_leaves()):
        assert i == leaf.get_leaf_index()
        data = leaf.get_data()[-100:]
        colors = [V[i] for _ in range(data.shape[0])]
        pyplot.scatter(x=data[:,0], y=data[:,1], c=colors, cmap=colormap, vmin=min(V), vmax=max(V), edgecolors='none')
    pyplot.colorbar()
    return


class QFunction(dict):
    
    def __init__(self, number_of_states, actions, **kwargs):
        super(QFunction, self).__init__(**kwargs)
        self.actions = actions
        for action in self.actions:
            self[action] = np.zeros(number_of_states)


    def split_state(self, index):
        for action in self.actions:
            self[action] = np.insert(Q[action], index, values=Q[action][index], axis=0)

            
    def get_action_values(self, state):
        return np.array([self[a][state] for a in self.actions])

    
    def update_reward(self, state, action, next_state, reward, model=None):
        
        max_q = max([Q[a][next_state] for a in self.actions])
        p = (reward + gamma * max_q - self[action][state])
        self[action][state] += alpha * p

        #        
        # prioritized sweep
        #
        if model is not None:

            # add last change to prioritized queue
            queue = PriorityQueue()
            if abs(p) > min_change:
                queue.add(p, state)
                
            P = model.get_transition_probabilities()
                
            # process queue for max. 1000 steps
            for _ in range(100):
                
                if queue.is_empty():
                    break
                
                # get Q value that changed most
                _, next_state = queue.pop()
                
                # do a sample backup for every potential predecessor (every 
                # state and action leading to s)
                # TODO: full backups instead
                for s in range(N):
                    
                    if s == next_state:
                        continue
                    
                    for a in self.actions: 
                    
                        if P[a][s,next_state] > 0.01:
                            
                            # simulate action
                            model.set_state(new_state=np.array([[s]]))
                            t, _, r = model.do_action(action=a)
                            t = t[0,0]
                            
                            # update Q value
                            max_q = max([Q[b][t] for b in self.actions])
                            p = (r + gamma * max_q - Q[a][s])
                            Q[a][s] += alpha * p
                            
                            # add s to queue if changed enough
                            if abs(p) > min_change:
                                queue.add(p, s)

        

if __name__ == '__main__':
    
    # extern model (maze)
    model_intern = None
    model_extern = env_maze.EnvMaze(seed=0)
    data, data_actions, _ = model_extern.do_random_steps(num_steps=1000)

    # worldmodel
    world_model = worldmodel.WorldModel()
    world_model.add_data(data=data, actions=data_actions)
    world_model.learn()
    
    # parameters for Q learning
    alpha = 0.5
    epsilon = 0.1
    gamma = 0.8
    min_change = 0.01
    N = world_model.get_number_of_states()
    Q = QFunction(number_of_states=N, actions=model_extern.get_available_actions())
    
    # explore!
    pyplot.ion()
    for t in range(10000):

        # get current state
        maze_state = model_extern.get_current_state()
        s = world_model.classify(maze_state)
        
        # learn
        if (t%100) == 0:
            
            split = world_model.single_splitting_step()
            
            # split value function
            if split is not None:
                index = split.node.children[0].get_leaf_index()
                Q.split_state(index=index)
                
            # (re-)build model_intern for new partition
            if split is not None or model_intern is None:
                N = world_model.get_number_of_states()
                state_rewards = world_model.get_gains()
                #maze_state = model_extern.get_current_state()
                s = world_model.classify(maze_state)
                model_intern = env_model.EnvModel(worldmodel=world_model, state_rewards=state_rewards, init_state=s)
                
            # plot
            pyplot.clf()
            pyplot.subplot(1, 2, 1)
            world_model.plot_data(color='last_gain', show_plot=False)
            pyplot.colorbar()
            pyplot.subplot(1, 2, 2)
            plot_value_function(world_model=world_model, Q=Q)
            pyplot.scatter(x=maze_state[0], y=maze_state[1], s=100)
            pyplot.draw()
        
        # epsilon-greedy action selection
        if np.random.random() < epsilon:
            selected_action = None
        else:
            action_values = Q.get_action_values(state=s)
            action_sum = np.sum(action_values)
            action_sum = action_sum**2
            if action_sum > 0:
                action_values /= action_sum
            else:
                action_values = np.ones(N) / N
            selected_action_index = action_values.cumsum().searchsorted(np.random.random()*np.sum(action_values))
            selected_action = model_extern.get_available_actions()[selected_action_index]
            
        # perform action
        new_maze_state, action, _ = model_extern.do_action(action=selected_action)
            
        # inform models about the transition
        world_model.add_data(new_maze_state, actions=[action])
        t = world_model.classify(new_maze_state)
        r = model_intern.set_state(new_state=t)
        
        Q.update_reward(state=s, action=action, next_state=t, reward=r, model=model_intern)

        
    print QtoV(Q)

    # plot
#     for i, a in enumerate(model_extern.get_available_actions()):
#         pyplot.subplot(2, 2, i+1)
#         plot_value_function(world_model=world_model, Q=Q, action=a)
#         pyplot.title(a)
    pyplot.show()
    