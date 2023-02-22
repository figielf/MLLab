from rl.dynamic_programming.mdp_deterministic_policy_optimization import optimize_mdp_policy_for_value_function
from tests.rl.dynamic_programming.gridworld_examples import build_windy_grid, build_standart_simple_grid, \
    build_negative_simple_grid, grid_5x5, build_windy_grid_no_wind, build_windy_grid_penalized
from tests.rl.dynamic_programming.iterative_policy_evaluation_tests import run_policy_evaluation

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

    grid = build_standart_simple_grid()
    #grid = build_negative_simple_grid()
    #grid = grid_5x5()

    #grid = build_windy_grid()
    #grid = build_windy_grid_no_wind()
    #grid = build_windy_grid_penalized()

    policy = policy
    #policy = probabilistic_policy

    V = run_policy_evaluation(grid, policy)

    new_policy = optimize_mdp_policy_for_value_function(grid, V)
    print('\ninitial policy:')
    grid.print_policy(policy)
    print('\noptimal policy for calculated value function V:')
    grid.print_policy(new_policy)
