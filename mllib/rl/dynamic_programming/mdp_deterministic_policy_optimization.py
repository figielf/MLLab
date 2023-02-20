from rl.dynamic_programming.iterative_policy_evaluation import get_game_probs


def optimize_mdp_policy_for_value_function_by_probs(game, state_transition_probs, reward_probs, V, gamma=0.9, trace_logs=False):
    optimal_policy = {}
    for state in game.actions.keys():  # will search for better Value for each state in the policy
        if not game.is_state_terminal(state):  # V(terminal_state) is always 0 so do not update
            best_value = float('-inf')
            best_action = None
            v_state = 0
            for action in game.ACTION_SPACE:  # calc Conditional Expectation over policy distribution pi(A|S)
                next_state_distribution = state_transition_probs.get((state, action), None)  # is action is allowed in the game (has probability to occur > 0)
                if next_state_distribution:
                    v_state = 0
                    for next_state in game.all_states():  # include reward and transition probability for each next state
                        rewards_distribution = reward_probs.get((state, action, next_state), {0.0: 1.0})
                        for r, p in rewards_distribution.items():
                            env_change_prob = next_state_distribution.get(next_state, 0) * p  # here we assume independece of the state transition and rewards distributions
                            dv = env_change_prob * (r + gamma * V[next_state])
                            v_state += dv

                    if v_state > best_value:
                        best_value = v_state
                        best_action = action
            optimal_policy[state] = best_action

    if trace_logs:
        game.print_policy(optimal_policy)

    return optimal_policy


def optimize_mdp_policy_for_value_function(game, V, gamma=0.9, trace_logs=False):
    state_transition_probs, reward_probs = get_game_probs(game)

    return optimize_mdp_policy_for_value_function_by_probs(game, state_transition_probs, reward_probs, V, gamma=gamma, trace_logs=trace_logs)


