import numpy as np

from rl.games.gridworld_simple import gridworld_simple
from rl.games.gridworld_windy import gridworld_windy


def calc_policy_probs(policy):
    probs = {}
    for state, actions in policy.items():
        if isinstance(actions, str):
            actions = {actions: 1.0}
        assert isinstance(actions, dict)
        for a, p in actions.items():
            probs[state, a] = p
    return probs


def get_game_probs(game):
    state_transition_probs = game.get_state_transition_probs()
    reward_probs = game.get_reward_probs()
    return state_transition_probs, reward_probs


def evaluate_policy_by_probs(game, policy_probs, state_transition_probs, reward_probs, V0=None, gamma=0.9, delta=1e-3, trace_logs=False):
    V = {}
    if V0:
        for s in game.all_states():
            V[s] = V0.get(s, 0)
    else:
        for s in game.all_states():
            V[s] = 0

    i = 0
    while True:
        biggest_diff = 0
        for state in game.all_states():  # will update Bellman equation for each state
            if not game.is_state_terminal(state):  # V(terminal_state) is always 0 so do not update
                old_value = V[state]
                v_state = 0
                for action in game.ACTION_SPACE:  # calc Conditional Expectation over policy distribution pi(A|S)
                    next_state_distribution = state_transition_probs.get((state, action), None)
                    if next_state_distribution:
                        for next_state in game.all_states():  # include reward and transition probability for each next state
                            rewards_distribution = reward_probs.get((state, action, next_state), {0.0: 1.0})
                            for r, p in rewards_distribution.items():
                                env_change_prob = next_state_distribution.get(next_state, 0) * p  # here we assume independece of the state transition and rewards distributions
                                dv = policy_probs.get((state, action), 0) * env_change_prob * (r + gamma * V[next_state])
                                v_state += dv

                biggest_diff = max(biggest_diff, np.abs(v_state - old_value))
                V[state] = v_state

        i += 1
        print(f'\nIteration {i}: max diff: {biggest_diff}')
        if trace_logs:
            game.print_values(V)
        if biggest_diff < delta:
            break
    return V


def evaluate_policy(game, policy, gamma=0.9, delta=1e-3, trace_logs=True):
    policy_probs = calc_policy_probs(policy)
    state_transition_probs, reward_probs = get_game_probs(game)

    return evaluate_policy_by_probs(game, policy_probs, state_transition_probs, reward_probs, gamma=gamma, delta=delta, trace_logs=trace_logs)
