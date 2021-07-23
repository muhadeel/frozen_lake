import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):

    value = np.zeros(env.n_states, dtype=np.float)
    delta = abs(theta)+1  # Force the loop entry
    i = 0
    while (delta > theta and i < max_iterations):
        delta = 0
        for state in range(env.n_states):
            value_old = value[state]
            value_tmp = 0
            for ns in range(env.n_states):
                value_tmp += env.p(ns, state, action=policy[state]) * (
                            (env.r(ns, state, action=policy[state])) + (gamma * value[ns]))

            value[state] = value_tmp
            delta = max(delta, np.abs(value_old - value[state]))
        i += 1
    return value

def policy_improvement(env, policy, value, gamma):
    improved_policy = np.zeros(env.n_states, dtype=int)
    policy_stable = True
    for state in range(env.n_states):
        old_action = policy[state]
        new_actions = []
        new_action_values = []
        for action in range(env.n_actions):
            for ns in range(env.n_states):
                new_actions.append(action)
                new_action_values.append(env.p(ns, state, action=action) * (
                            (env.r(ns, state, action=action)) + (gamma * value[ns])))
        best_action = new_actions[new_action_values.index(max(new_action_values))]
        improved_policy[state] = best_action
        if old_action != best_action:  # and state != env.absorbing_state_idx:
            policy_stable = False

    return improved_policy, policy_stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    while True:
        value = policy_evaluation(env,policy,gamma,theta,max_iterations)
        policy,policy_stable = policy_improvement(env,policy,value,gamma)
        if policy_stable:
            break

    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    policy = np.zeros(env.n_states, dtype = int)

    delta = abs(theta)+1  # Force the loop entry
    i = 0
    while (delta > theta and i < max_iterations):
        delta = 0
        for state in range(env.n_states):
            value_old = value[state]
            value_new = []
            for action in range(env.n_actions):
                value_tmp = 0
                for ns in range(env.n_states):
                    value_tmp += env.p(ns, state, action=action) * (
                        (env.r(ns, state, action=action)) + (gamma * value[ns]))
                value_new.append(value_tmp)

            value[state] = max(value_new)
            delta = max(delta, np.abs(value_old - value[state]))
        i += 1

    for state in range(env.n_states):
        new_actions = []
        new_action_values = []
        for action in range(env.n_actions):
            for ns in range(env.n_states):
                new_actions.append(action)
                new_action_values.append(env.p(ns, state, action=action) * (
                            (env.r(ns, state, action=action)) + (gamma * value[ns])))
        best_action = new_actions[new_action_values.index(max(new_action_values))]
        policy[state] = best_action

    return policy, value
