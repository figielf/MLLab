import numpy as np

from rl.games.play_grid import play_one_move_by_optimal_action_based_on_q
from rl.games.epsilon_greedy import get_best_action_and_q


def temporal_difference_q_learning_policy_evaluation(game_factory, n_episodes=10000, gamma=0.9, alpha=0.1, eps=0.1):
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
            _, max_q = get_best_action_and_q(Q[new_state])

            old_q = Q[state][action]
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * max_q - Q[state][action])
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