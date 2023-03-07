import numpy as np


def get_best_action_and_q(action_q):
    best_q = max(action_q.values())
    best_actions = [a for a, q in action_q.items() if q == best_q]
    return np.random.choice(best_actions), best_q


def get_optimal_action_from_q(Q, state):
    action, _ = get_best_action_and_q(Q[state])
    return action


def get_epsilon_greedy_policy_action(action_space, policy, state, eps):
    action = policy[state]

    # explore by epsilon greedy strategy
    p = np.random.random()
    if p < eps:
        action = np.random.choice(action_space)
    return action


def get_epsilon_greedy_optimal_action_from_q(action_space, Q, state, eps):
    action = get_optimal_action_from_q(Q, state)

    # explore by epsilon greedy strategy
    p = np.random.random()
    if p < eps:
        action = np.random.choice(action_space)
    return action

