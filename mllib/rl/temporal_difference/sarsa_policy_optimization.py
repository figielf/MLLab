import numpy as np

from rl.monte_carlo.play_grid import get_epsilon_greedy_optimal_action_from_q, \
    play_one_move_by_optimal_action_based_on_q, get_best_action_and_q


def temporal_difference_sarsa_policy_evaluation(game_factory, n_episodes=10000, gamma=0.9, alpha=0.1, eps=0.1):
    Q_counts = {}
    Q = {}  # dict[state, action] = p(s', r | s, a)
    game_example = game_factory()
    for st in game_example.all_states():
        Q[st] = {}
        for ac in game_example.ACTION_SPACE:
            Q_counts[(st, ac)] = 0
            Q[st][ac] = 0

    deltas = []
    episode_rewards = []
    for i in range(n_episodes):
        if i % 1000 == 0:
            print(f'episode {i}/{n_episodes}')
        game = game_factory()
        state = game.get_current_state()
        delta = 0
        episode_reward = 0
        while not game.is_game_over():
            action, reward, new_state = play_one_move_by_optimal_action_based_on_q(game, Q, on_invalid_action='no_effect', with_epsilon_greedy=True, eps=eps)
            next_action = get_epsilon_greedy_optimal_action_from_q(game, Q, new_state, eps=eps)

            old_q = Q[state][action]
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[new_state][next_action] - Q[state][action])
            Q_counts[(state, action)] += 1
            episode_reward += reward

            # calculate change in Q values
            delta = max(delta, np.abs(Q[state][action] - old_q))  # check convergence

            state = new_state
        deltas.append(delta)
        episode_rewards.append(episode_reward)
    history = deltas, episode_rewards

    V = {}
    target_policy = {}
    for state, actions_q in Q.items():
        best_q_action, V[state] = get_best_action_and_q(actions_q)
        if not game.is_state_terminal(state):
            target_policy[state] = best_q_action

    return target_policy, V, Q, history, Q_counts