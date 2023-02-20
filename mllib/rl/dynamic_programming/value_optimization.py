import numpy as np

from rl.dynamic_programming.iterative_policy_evaluation import get_game_probs


def calculate_optimized_value_by_probs(game, state_transition_probs, reward_probs, gamma=0.9, delta=1e-3, trace_logs=False):
    V = {}
    for s in game.all_states():
        V[s] = 0

    i = 0
    while True:
        biggest_diff = 0
        for state in game.all_states():  # will update Bellman equation for each state
            if not game.is_state_terminal(state):  # V(terminal_state) is always 0 so do not update
                old_value = V[state]
                best_value = float('-inf')
                for action in game.ACTION_SPACE:  # calc Conditional Expectation over policy distribution pi(A|S)
                    next_state_distribution = state_transition_probs.get((state, action), None)
                    if next_state_distribution:
                        v = 0
                        for next_state in game.all_states():  # include reward and transition probability for each next state
                            rewards_distribution = reward_probs.get((state, action, next_state), {0.0: 1.0})
                            for r, p in rewards_distribution.items():
                                env_change_prob = next_state_distribution.get(next_state, 0) * p  # here we assume independece of the state transition and rewards distributions
                                dv = env_change_prob * (r + gamma * V[next_state])
                                v += dv
                        best_value = max(best_value, v)

                biggest_diff = max(biggest_diff, np.abs(best_value - old_value))
                V[state] = best_value

        i += 1
        print(f'\nIteration {i}: max diff: {biggest_diff}')
        if trace_logs:
            game.print_values(V)

        if biggest_diff < delta:
            break
    return V


def calculate_optimized_value(game, gamma=0.9, trace_logs=False):
    state_transition_probs, reward_probs = get_game_probs(game)

    return calculate_optimized_value_by_probs(game, state_transition_probs, reward_probs, gamma=gamma, trace_logs=trace_logs)
