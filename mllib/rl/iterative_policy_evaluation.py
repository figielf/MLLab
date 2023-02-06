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


def _calc_prod_ORG(grid):
    def get_next_state(grid, s, a):
        # this answers: where would I end up if I perform action 'a' in state 's'?
        i, j = s

        # if this action moves you somewhere else, then it will be in this dictionary
        if a in grid.actions[s]:
            if a == 'U':
                i -= 1
            elif a == 'D':
                i += 1
            elif a == 'R':
                j += 1
            elif a == 'L':
                j -= 1
        return i, j


    transition_probs = {}
    rewards = {}
    for i in range(grid.n_rows):
        for j in range(grid.n_cols):
            s = (i, j)
            if not grid.is_state_terminal(s):
                for a in grid.ACTION_SPACE:
                    s2 = get_next_state(grid, s, a)
                    transition_probs[(s, a)] = {s2: 1}
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]
    return transition_probs, rewards


def evaluate_deterministic_policy(policy, game, gamma=0.9, delta=1e-3, print_values=False):
    # we assume rewards are deterministic
    V = {}
    for s in game.all_states():
        V[s] = 0

    policy_probs = calc_policy_probs(policy)
    if isinstance(game, gridworld_simple):
        state_transition_probs = game.get_state_transition_probs()
        rewards = game.get_reward_probs()
    elif isinstance(game, gridworld_windy):
        state_transition_probs = game.transition_probs
        rewards = game.get_reward_probs()

    i = 0
    while True:
        biggest_diff = 0
        for state in game.all_states():  # will update Bellman equation for each state
            if not game.is_state_terminal(state):  # V(terminal_state) is always 0 so do not update
                old_value = V[state]
                v_state = 0
                for action in game.ACTION_SPACE:  # calc Conditional Expectation over policy distribution pi(A|S)
                    p_trans = state_transition_probs.get((state, action), None)
                    if p_trans:
                        for next_state in game.all_states():  # include reward and transition probability for each next state
                            rewards_distribution = rewards.get((state, action, next_state), {0.0: 1.0})
                            for r, p in rewards_distribution.items():
                                env_distrib = p_trans.get(next_state, 0) * p  # here we assume independece of the state transition and rewards distributions
                                dv = policy_probs.get((state, action), 0) * env_distrib * (r + gamma * V[next_state])
                                v_state += dv

                biggest_diff = max(biggest_diff, np.abs(v_state - old_value))
                V[state] = v_state

        i += 1
        print(f'\nIteration {i}: max diff: {biggest_diff}')
        if print_values:
            game.print_values(V)
        if biggest_diff < delta:
            break
    return V
