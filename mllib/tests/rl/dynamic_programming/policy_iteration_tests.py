import numpy as np

from rl.dynamic_programming.policy_iteration import optimize_policy_by_policy_iteration
from tests.rl.dynamic_programming.gridworld_examples import grid_5x5, build_windy_grid_penalized, \
    build_standart_simple_grid, build_negative_simple_grid, build_windy_grid, build_windy_grid_no_wind

if __name__ == '__main__':
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L'}

    probabilistic_policy = {
        (2, 0): {'U': 0.5, 'R': 0.5},
        (1, 0): {'U': 1.0},
        (0, 0): {'R': 1.0},
        (0, 1): {'R': 1.0},
        (0, 2): {'R': 1.0},
        (1, 2): {'U': 1.0},
        (2, 1): {'R': 1.0},
        (2, 2): {'U': 1.0},
        (2, 3): {'L': 1.0},
    }

    odd_policy = {
        (0, 0): 'R',
        (0, 1): 'U',
        (0, 2): 'D',
        (1, 0): 'U',
        (1, 2): 'D',
        (2, 0): 'L',
        (2, 1): 'R',
        (2, 2): 'L',
        (2, 3): 'D'}

    #grid = build_standart_simple_grid()
    #grid = build_negative_simple_grid()
    #grid = grid_5x5()

    #grid = build_windy_grid()
    #grid = build_windy_grid_no_wind()
    grid = build_windy_grid_penalized(-0.2)

    random_policy = {}
    for s in grid.actions.keys():
        random_policy[s] = np.random.choice(grid.ACTION_SPACE)

    #policy = policy
    #policy = probabilistic_policy
    policy = random_policy

    optimal_policy, V = optimize_policy_by_policy_iteration(game=grid, initial_policy=odd_policy, gamma=0.9, trace_logs=False)
    print('\n\n----------OPTIMIZATION OVER----------')
    print('\ninitial policy:')
    grid.print_policy(policy)
    print('\nfinal V:')
    grid.print_values(V)
    print('\nFinal policy (optimal policy for calculated value function V):')
    grid.print_policy(optimal_policy)
