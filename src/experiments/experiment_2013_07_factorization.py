import numpy as np
import scipy.linalg
import scipy.spatial

from matplotlib import pyplot

import studienprojekt.env_cube

import worldmodel


def get_transition_graph(model, fast_action, k=15, normalize=True):

    refs = model._get_data_refs()
    data = model._get_data_for_refs(refs)
    actions = model._get_actions_for_refs(refs)
    N = len(refs)        
    
    # pairwise distances
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)

    # transition matrix
    W = np.zeros((N, N))
    #W += 1e-8
    
    # number of actions
    action_set = set(actions)
    if None in action_set:
        action_set.remove(None)
    n_actions = len(action_set)
    weight = n_actions - 1

    # transitions to neighbors
    # s - current node
    # t - neighbor node
    # u - following node
    for s, _ in enumerate(refs):
        indices = np.argsort(distances[s])  # closest one should be the point itself
        for t in indices[0:k+1]:
            if s == t:
                continue
            if actions[t] == fast_action:
                W[s,t] = 1
                W[t,s] = 1

    # transitions to successors
    # s - current node
    # t - neighbor node
    # u - following node (of neighbor)
    for s, _ in enumerate(refs):
        indices = np.argsort(distances[s])  # closest one should be the point itself
        for t in indices[0:k+1]:
            ref_t = refs[t]
            ref_u = ref_t + 1
            if ref_u not in refs:
                continue
            u = refs.index(ref_u)
            if actions[u] == fast_action:
                W[s,u] = -weight
                W[u,s] = -weight
            else:
                W[s,u] = 1
                W[u,s] = 1
                
    # normalize matrix
    if normalize:
        d = np.sum(W, axis=1)
        for i in range(N):
            if d[i] > 0:
                W[i] = W[i] / d[i]
            
    return W, refs, data, actions
    
    
if __name__ == '__main__':
    
    # environment
    #env = studienprojekt.env_cube.EnvCube(step_size=0.2, sigma=0.01)
    env = studienprojekt.env_cube.EnvCube(step_size=0.1, sigma=0.05)
    data, actions = env.do_random_steps(num_steps=2000)
    
    # train model
    model = worldmodel.WorldModelFactorize()
    model.add_data(data=data, actions=actions)
    model.learn()
    #model.single_splitting_step(action=None, min_gain=0.0)
    
    for i, (a, m) in enumerate(model.models.items()):
        mi = worldmodel.WorldModel._mutual_information(m.transitions[a])
        pyplot.subplot(2, 2, (i+1))
        pyplot.title(mi)
        m.plot_states(show_plot=False)
        #m.plot_tree_data(color='none', show_plot=False)
        
#     m = m.get_leaves()[0]
#     W, refs, data, actions = get_transition_graph(m, fast_action=a, k=5, normalize=False)
#     N = W.shape[0]
#     for u in range(N):
#         for v in range(u+1, N):
#             if W[u,v] > 0:
#                 pyplot.plot([data[u][0], data[v][0]], [data[u][1], data[v][1]], '-', c='blue')
#             if W[u,v] < 0:
#                 pyplot.plot([data[u][0], data[v][0]], [data[u][1], data[v][1]], '-', c='red')
                
    pyplot.show()
    