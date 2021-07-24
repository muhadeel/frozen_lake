import numpy as np
from numpy.random import default_rng

def e_greedy(epsilon, q, actions):
    
    rng = default_rng()
    noise = rng.normal(loc=0.0, scale=1e-10, size=4)
    q = np.add(q,noise)

    if rng.uniform(0,1) < (1-epsilon):
        return q.argmax()
    else:
        return rng.choice(actions)

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        
        s = env.reset()
        
        a = e_greedy(epsilon[i], q[s], env.n_actions)
        
        done = False
        
        while not done:
            
            s2, r, done = env.step(a)
            a2 = e_greedy(epsilon[i], q[s2], env.n_actions)
            
            q[s, a] += eta[i] * (r + (gamma * q[s2, a2]) - q[s, a])

            s = s2
            a = a2

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        
        done = False
        
        while not done:
            
            a = e_greedy(epsilon[i], q[s], env.n_actions)
            s2, r, done = env.step(a)
            
            q[s, a] += eta[i] * (r + (gamma * max(q[s2])) - q[s, a])

            s = s2
    

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value