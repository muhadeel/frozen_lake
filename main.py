from environments.frozen_lake_environment import FrozenLake
from rl_algorithms.non_tabular_model_free import LinearWrapper, linear_sarsa, linear_q_learning
from rl_algorithms.tabular_model_based import policy_iteration, value_iteration
from rl_algorithms.tabular_model_free import sarsa, q_learning
from rl_algorithms.tabular_model_based import policy_evaluation,policy_improvement  #test, remove when done

def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    # play(env)

    # ------Test Pol Eval---------#
    # print("EVALUATE INITIAL POLICY")
    # gamma = 0.9
    # theta = 0.001
    # max_iterations = 100
    # polTest = [3,3,2,1,2,2,2,2,3,3,2,2,2,3,3,3,3]
    # value = policy_evaluation(env,polTest,gamma, theta, max_iterations)
    # env.render(polTest,value)
    #-----------------------------#

    # ------Test Pol Improvement---------#
    # print("\nIMPROVE POLICY")
    # gamma = 0.9
    # theta = 0.001
    # max_iterations = 100
    # polTest = [3,3,2,1,2,2,2,2,3,3,2,2,2,3,3,3,3]
    # new_pol = policy_improvement(env,polTest,value,gamma)
    # print("new policy:", new_pol)
    # new_value = policy_evaluation(env,new_pol[0],gamma, theta, max_iterations)
    # env.render(new_pol[0],new_value)
    #-----------------------------#


    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('')

    print('## Policy iteration')
    # polTest = [3, 3, 2, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 3, 3, 3, 3]
    # policy, value = policy_iteration(env, gamma, theta, max_iterations,polTest)
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    #
    # print('')
    #
    # print('## Value iteration')
    # policy, value = value_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('# Model-free algorithms')
    # max_episodes = 2000
    # eta = 0.5
    # epsilon = 0.5
    #
    # print('')
    #
    # print('## Sarsa')
    # policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Q-learning')
    # policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # linear_env = LinearWrapper(env)
    #
    # print('## Linear Sarsa')
    #
    # parameters = linear_sarsa(linear_env, max_episodes, eta,
    #                           gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)
    #
    # print('')
    #
    # print('## Linear Q-learning')
    #
    # parameters = linear_q_learning(linear_env, max_episodes, eta,
    #                                gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)


def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid Action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))


if __name__ == '__main__':
    main()