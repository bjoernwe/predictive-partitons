import numpy as np

import mdp


class Environment(object):
    '''
    Base class for all environments.
    
    To implement a subclass you need to initialize the list of possible actions
    (integers) and set the environment to some initial state vector. Also, the 
    _do_action method has to be implemented by the subclass. If you need a 
    random generator, use self.rnd if possible because it has a seed.
    '''


    def __init__(self, seed=None):
        """
        Initializes the environment including an initial state.
        """
        self.ndim = None                # initialize in sub-class
        self.actions_dict = {0: None}   # initialize in sub-class
        self.current_state = None       # initialize in sub-class
        self.last_action = None
        self.last_reward = None
        self.rnd = np.random.RandomState(seed)
            
            
    def get_number_of_possible_actions(self):
        """
        Returns the number N for the possible actions 0, ..., N-1
        """
        return len(self.actions_dict)
    

    def get_actions_dict(self):
        """
        Returns a dictionary of actions
        """
        return self.actions_dict
    
    
    def get_current_state(self):
        """
        Returns the current state of the environment.
        """
        return self.current_state
    
    
    def get_last_action(self):
        """
        Returns the last action performed.
        """
        return self.last_action
    
    
    def get_last_reward(self):
        """
        Returns the last reward received in the previous step.
        """
        return self.last_reward
    
        
    def do_random_steps(self, num_steps=1):
        """
        Performs random actions and returns a three results:
        1) a matrix containing the resulting states
        2) a vector of actions performed, one shorter than the state vector 
           because there is no action for the last state yet
        3) a vector containing the rewards received in each step
        """
        
        rewards = np.zeros(num_steps-1)
        states = np.zeros((num_steps, self.ndim))
        states[0] = self.get_current_state()
        
        if self.actions_dict is None:
            random_actions = None
            for i in range(num_steps-1):
                states[i+1], _, rewards[i] = self.do_action(action=None) 
        else:
            num_actions = self.get_number_of_possible_actions()
            random_actions = self.rnd.randint(0, high=num_actions, size=num_steps-1)
            for i, action in enumerate(random_actions):
                states[i+1], _, rewards[i] = self.do_action(action=action) 
                
        assert len(states) == len(rewards) + 1
        assert random_actions is None or len(states) == len(random_actions) + 1
        return [states, random_actions, rewards]

    
    def do_action(self, action=None):
        """
        Performs the given action and returns the resulting state, the action
        and the received reward. If no action is given, a random action is
        selected.
        """
        
        # select random action
        if action is None and self.actions is not None:
            action = self.rnd.choice(self.actions)

        # perform action        
        self.current_state, reward = self._do_action(action=action)
        self.last_action = action
        self.last_reward = reward
        
        return self.current_state, self.last_action, self.last_reward
        

    def _do_action(self, action):
        """
        Performs the given action and returns the resulting state as well as 
        some reward value.
        """
        raise RuntimeError('method not implemented yet')
    
    
    def generate_training_data(self, num_steps, noisy_dims=0, whitening=True, expansion=1, chunks=1):
        """
        Generates a list of data chunks. Each chunks is a 3-tuple of generated
        data, corresponding actions and reward values/labels. 
        """
        
        # for every chunk ...
        result = []
        for _ in range(chunks):

            # data
            data, actions, rewards = self.do_random_steps(num_steps=num_steps)
            
            # make sure data has two dimensions
            if data.ndim == 1:
                data = np.array(data, ndmin=2).T 
    
            # add noisy dim
            for _ in range(noisy_dims):
                noise_complete = 1. * self.rnd.rand(num_steps)
                data = np.insert(data, data.shape[1], axis=1, values=noise_complete)
    
            # expansion
            expansion_node = mdp.nodes.PolynomialExpansionNode(degree=expansion)
            data = expansion_node.execute(data)
    
            # whitening
            if whitening:
                whitening_node = mdp.nodes.WhiteningNode()
                whitening_node.train(data)
                data = whitening_node.execute(data)
    
            result.append((data, actions, rewards))
            
        return result
