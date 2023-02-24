import numpy as np

from rl.monte_carlo.play_grid import play_one_move_by_deterministic_policy


def temporal_difference_value_evaluation(game_factory, policy, n_episodes=10000, gamma=0.9, alpha=0.1):
    V = {}
    game_example = game_factory()
    for s in game_example.all_states():
        V[s] = 0

    history = []
    for i in range(n_episodes):
        game = game_factory()
        state = game.get_current_state()
        delta = 0
        while not game.is_game_over():
            old_V = V[state]
            action, reward, new_state = play_one_move_by_deterministic_policy(game, policy, on_invalid_action='no_effect', with_epsilon_greedy=True)

            V[state] = V[state] + alpha * (reward + gamma * V[new_state] - V[state])

            delta = max(delta, np.abs(V[state] - old_V))
            state = new_state
        history.append(delta)
    return V, history
