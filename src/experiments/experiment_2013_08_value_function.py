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
        
        
def plot_value_function(world_model, Q):
    """
    Plots data of worldmodel colored with value function from Q.
    """
    V = QtoV(Q)
    colormap = pyplot.cm.get_cmap('summer')
    for i, leaf in enumerate(world_model.tree.get_leaves()):
        assert i == leaf.get_leaf_index()
        data = leaf.get_data()
        colors = [V[i] for _ in range(data.shape[0])]
        pyplot.scatter(x=data[:,0], y=data[:,1], c=colors, cmap=colormap, vmin=min(V), vmax=max(V), edgecolors='none')
    pyplot.colorbar()
    return

        

if __name__ == '__main__':
    
    model = None
    maze = env_maze.EnvMaze(seed=0)
    data, actions, _ = maze.do_random_steps(num_steps=100)

    world_model = worldmodel.WorldModel()
    world_model.add_data(data=data, actions=actions)
    world_model.learn()
    
    alpha = 0.5
    epsilon = 0.1
    gamma = 0.9
    min_change = 0.01
    N = world_model.get_number_of_states()
    Q = {}
    
    #T = world_model.transitions
    #state_rewards = world_model.get_last_gains()
    #s = world_model.classify(maze.get_current_state())
    #model = env_model.EnvModel(transitions=T, state_rewards=state_rewards, init_state=s)
    #print state_rewards

    # initialize Q
    for action in maze.get_available_actions():
        Q[action] = np.zeros(N)
    
    for i in range(2000):

        # get current state
        maze_state = maze.get_current_state()
        s = world_model.classify(maze_state)
        
        # init priority queue
        queue = PriorityQueue()
        
        # learn
        if (i%100) == 0:
            
            split = world_model.single_splitting_step()
            if split is not None:
                # split value function
                index = split.node.children[0].get_leaf_index()
                print index
                for action in maze.get_available_actions():
                    Q[action] = np.insert(Q[action], index, values=Q[action][index], axis=0)
                queue.add(np.float('inf'), index)
                queue.add(np.float('inf'), index+1)
                
            # (re-)build model for new partition
            if split is not None or model is None:
                N = world_model.get_number_of_states()
                T = world_model.get_transitions(copy=True)
                state_rewards = world_model.get_last_gains()
                maze_state = maze.get_current_state()
                s = world_model.classify(maze_state)
                model = env_model.EnvModel(transitions=T, state_rewards=state_rewards, init_state=s)
        
        # epsilon-greedy action selection
        if np.random.random() < epsilon:
            new_maze_state, a, _ = maze.do_action(action=None)
        else:
            print Q
            print s
            action_values = np.array([Q[a][s] for a in maze.get_available_actions()])
            action_sum = np.sum(action_values)
            if action_sum > 0:
                action_values /= action_sum
            else:
                action_values = np.ones(N) / N
        #    a = max(maze.get_available_actions(), key=lambda x: Q[x][s])
            new_maze_state, a, _ = maze.do_action(action=a)
            
        # inform models about the transition
        world_model.add_data(new_maze_state, actions=[a])
        t = world_model.classify(new_maze_state)
        r = model.add_transition(target_state=t, action=a, previous_state=s)
        
        # update Q value
        max_q_t = max([Q[b][t] for b in maze.get_available_actions()])
        p = (r + gamma * max_q_t - Q[a][s])
        Q[a][s] += alpha * p

        #        
        # prioritized sweep
        #
        if abs(p) > min_change:
            queue.add(p, s)
            
        # process queue for max. 1000 steps
        P = model.get_transition_probabilities()
        for _ in range(1000):
            
            if queue.is_empty():
                break
            
            # get Q value that changed most
            _, s = queue.pop()
            
            # do a sample backup for every potential predecessor
            t = s
            for s in range(world_model.get_number_of_states()):
                
                # any action leading from state s to t?
                greedy_action = max(maze.get_available_actions(), key=lambda x: P[x][s,t])
                
                # perform greedy action and update Q
                if P[greedy_action][s,t] > 0.01:
                    
                    # simulate action
                    model.set_state(new_state=np.array([[s]]))
                    t, a, r = model.do_action(action=greedy_action)
                    t = t[0,0]
                    
                    # update Q value
                    max_q_t = max([Q[b][t] for b in maze.get_available_actions()])
                    p = (r + gamma * max_q_t - Q[a][s])
                    Q[a][s] += alpha * p
                    
                    # add s to queue if changed enough
                    if abs(p) > min_change:
                        queue.add(p, s)

    print QtoV(Q)

    # plot
    pyplot.subplot(1, 2, 1)            
    world_model.plot_data(color='last_gain', show_plot=False)
    pyplot.colorbar()
    pyplot.subplot(1, 2, 2)
    plot_value_function(world_model, Q)
    pyplot.show()
    