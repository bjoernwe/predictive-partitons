import numpy as np

from matplotlib import pyplot

import worldmodel

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
        


class QFunction(dict):
    
    def __init__(self, number_of_states, actions, alpha=.5, gamma=.8, min_change=.01, **kwargs):
        super(QFunction, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.min_change = min_change
        self.number_of_states = number_of_states
        self.actions = actions
        for action in self.actions:
            self[action] = np.zeros(number_of_states)


    def split_state(self, index):
        for action in self.actions:
            self[action] = np.insert(self[action], index, values=self[action][index], axis=0)
        self.number_of_states += 1
        return

            
    def get_action_values(self, state):
        return np.array([self[a][state] for a in self.actions])
    
    
    def get_max_action_value(self, state):
        return np.max(self.get_action_values(state=state))

    
    def update_reward_sample(self, state, action, next_state, reward, model=None):
        
        N = self.number_of_states
        
        max_q = max([self[a][next_state] for a in self.actions])
        p = (reward + self.gamma * max_q - self[action][state])
        self[action][state] += self.alpha * p

        #        
        # prioritized sweep
        #
        if model is not None:

            # add last change to prioritized queue
            queue = PriorityQueue()
            if abs(p) > self.min_change:
                queue.add(p, state)
                
            # process queue for max. 1000 steps
            for _ in range(100):
                
                if queue.is_empty():
                    break
                
                # get Q value that changed most
                _, next_state = queue.pop()
                
                # do a sample backup for every potential predecessor (every 
                # state and action leading to s)
                for s in range(N):
                    
                    if s == next_state:
                        continue
                    
                    for a in self.actions: 
                    
                        if model.P[a][s,next_state] > 0.01:
                            
                            # simulate action
                            model.set_state(new_state=np.array([[s]]))
                            t, _, r = model.do_action(action=a)
                            t = t[0,0]
                            
                            # update Q value
                            max_q = max([self[b][t] for b in self.actions])
                            p = (r + self.gamma * max_q - self[a][s])
                            self.Q[a][s] += self.alpha * p
                            
                            # add s to queue if changed enough
                            if abs(p) > self.min_change:
                                queue.add(p, s)


    def update_reward_full(self, state, action, model=None):
        
        N = self.number_of_states
        P = model.P
        R = model.R
        
        # do a full backup
        s = state
        a = action
        new_value = np.sum([P[a][s,t] * (R[a][s,t] + self.gamma * self.get_max_action_value(t)) for t in range(N)])
        p = new_value - self[a][s]
        self[a][s] = new_value 
                
        #        
        # prioritized sweep
        #
        if model is not None:

            # add last change to prioritized queue
            queue = PriorityQueue()
            if abs(p) > self.min_change:
                queue.add(p, state)
                
            # process queue for max. 1000 steps
            for _ in range(100):
                
                if queue.is_empty():
                    break
                
                # get Q value that changed most
                _, next_state = queue.pop()
                
                # do a full backup for every potential predecessor (every 
                # state and action leading to s)
                for s in range(N):
                    
                    if s == next_state:
                        continue
                    
                    for a in self.actions: 
                    
                        if P[a][s,next_state] > 0.01:

                            new_value = np.sum([P[a][s,t] * (R[a][s,t] + self.gamma * self.get_max_action_value(t)) for t in range(N)])
                            p = new_value - self[a][s]
                            self[a][s] = new_value 
                            
                            # add s to queue if changed enough
                            if abs(p) > self.min_change:
                                queue.add(p, s)

        

class RLExploration(object):
    
    def __init__(self, model):
        
        # extern model (maze)
        self.model_extern = model
        self.model_intern = None
        self.data, self.data_actions, _ = model.do_random_steps(num_steps=1000)
    
        # worldmodel
        self.world_model = worldmodel.WorldModel()
        self.world_model.add_data(data=self.data, actions=self.data_actions)
        self.world_model.learn()
        
        # parameters for Q learning
        self.epsilon = 0.1
        self.N = self.world_model.get_number_of_states()
        self.Q = QFunction(number_of_states=self.N, actions=model.get_available_actions())
        
        
    def explore(self, steps=1, live_plot=True):
    
        if live_plot:
            pyplot.ion()
            
        # explore!
        for t in range(steps):
    
            # get current state
            maze_state = self.model_extern.get_current_state()
            s = self.world_model.classify(maze_state)
            
            # learn
            if (t%100) == 0:
                
                split = self.world_model.single_splitting_step()
                
                # split value function
                if split is not None:
                    index = split.node._children[0].get_leaf_index()
                    self.Q.split_state(index=index)
                    
                # (re-)build model_intern for new partition
                if split is not None or self.model_intern is None:
                    self.N = self.world_model.get_number_of_states()
                    s = self.world_model.classify(maze_state)
                    self.model_intern = env_model.EnvModelIntrinsic(worldmodel=self.world_model, init_state=s)
                    #model_intern = env_model.EnvModelExtrinsic(worldmodel=self.world_model, init_state=s)
                    
                # plot
                if live_plot:
                    pyplot.clf()
                    #pyplot.subplot(1, 2, 1)
                    self.world_model.plot_data(color='last_gain', show_plot=False)
                    pyplot.colorbar()
                    #pyplot.subplot(1, 2, 2)
                    #plot_data_with_value(world_model=self.world_model, Q=self.Q)
                    #pyplot.scatter(x=maze_state[0], y=maze_state[1], s=100)
                    pyplot.draw()
            
            # epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                selected_action = None
            else:
                action_values = self.Q.get_action_values(state=s)
                action_sum = np.sum(action_values)
                action_sum = action_sum**2
                if action_sum > 0:
                    action_values /= action_sum
                else:
                    A = len(action_values)
                    action_values = np.ones(A) / A
                selected_action_index = action_values.cumsum().searchsorted(np.random.random()*np.sum(action_values))
                selected_action = self.model_extern.get_available_actions()[selected_action_index]
                
            # perform action
            new_maze_state, action, _ = self.model_extern.do_action(action=selected_action)
                
            # inform models about the transition
            self.world_model.add_data(new_maze_state, actions=[action])
            t = self.world_model.classify(new_maze_state)
            r = self.model_intern.set_state(new_state=t, action=action)
            
            #Q.update_reward_sample(state=s, action=action, next_state=t, reward=r, model=model_intern)
            self.Q.update_reward_full(state=s, action=action, model=self.model_intern)
    


def QtoV(Q):
    """
    Calculates a value function from Q (assuming greed strategy).
    """
    N = Q.itervalues().next().shape[0]
    V = np.zeros(N)
    for a in Q.keys():
        V = np.maximum(V, Q[a])
    return V
        

        
def plot_data_with_value(world_model, Q, action=None):
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
        data = leaf.get_data()[:]
        colors = [V[i] for _ in range(data.shape[0])]
        pyplot.scatter(x=data[:,0], y=data[:,1], c=colors, cmap=colormap, vmin=min(V), vmax=max(V), edgecolors='none')
    pyplot.colorbar()
    return

        