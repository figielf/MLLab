from rl.dynamic_programming.iterative_policy_evaluation import get_game_probs, evaluate_policy_by_probs, calc_policy_probs
from rl.dynamic_programming.mdp_deterministic_policy_optimization import optimize_mdp_policy_for_value_function_by_probs


def policy_compare(p1, p2):
    # compares policies of type dict[tuple(int, int), str] (dict[state, action])
    for s, a in p1.items():
        if s not in p2.keys():

            return False
        if p2[s] != a:
            return False
    return True


def optimize_policy_by_policy_iteration_by_probs(game, policy, state_transition_probs, reward_probs, gamma=0.9, trace_logs=False):
    V = None
    print('\nInitial policy:')
    game.print_policy(policy)
    iter = 1
    while True:
        policy_probs = calc_policy_probs(policy)
        V = evaluate_policy_by_probs(game, policy_probs, state_transition_probs, reward_probs, V0=V, gamma=gamma, delta=1e-3, trace_logs=trace_logs)
        new_policy = optimize_mdp_policy_for_value_function_by_probs(game, state_transition_probs, reward_probs, V, gamma=gamma, trace_logs=trace_logs)
        policy_converged = policy_compare(policy, new_policy)
        print(f'\n\n### Iteration {iter} ###')
        print('\nvalue:')
        game.print_values(V)
        print('\npolicy:')
        game.print_policy(new_policy)
        iter += 1
        if policy_converged:
            break
        policy = new_policy
    return new_policy, V


def optimize_policy_by_policy_iteration(game, initial_policy, gamma=0.9, trace_logs=False):
    state_transition_probs, reward_probs = get_game_probs(game)

    return optimize_policy_by_policy_iteration_by_probs(game, initial_policy, state_transition_probs, reward_probs, gamma=gamma, trace_logs=trace_logs)
