import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rl.games.grid_policies import generate_random_grid_policy
from rl.monte_carlo.epsilon_greedy_policy_optimization import \
    monte_carlo_epsilon_greedy_deterministic_policy_optimization
from tests.rl.dynamic_programming.gridworld_examples import grid_5x5, build_windy_grid_penalized, \
    build_standart_simple_grid, build_negative_simple_grid, build_windy_grid, build_windy_grid_no_wind
from tests.rl.monte_carlo.exploring_starts_policy_optimization_tests import print_state_visit_counts

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

    #grid_factory=lambda: build_standart_simple_grid()
    grid_factory=lambda: build_negative_simple_grid()
    #grid_factory=lambda: grid_5x5()

    #grid_factory=lambda: build_windy_grid(quiet=True)
    #grid_factory=lambda: build_windy_grid_no_wind(quiet=True)
    #grid_factory=lambda: build_windy_grid_penalized(-0.2, quiet=True)

    grid_example = grid_factory()
    random_policy = generate_random_grid_policy(grid_example)

    #policy = policy
    #policy = probabilistic_policy
    policy = random_policy
    print('\ninitial policy:')
    grid_example.print_policy(policy)

    optimal_policy, V, Q, history, Q_counts = monte_carlo_epsilon_greedy_deterministic_policy_optimization(game_factory=grid_factory, initial_policy=random_policy, gamma=0.9)
    print('\n\n----------OPTIMIZATION OVER----------')
    print('\nFinal policy (optimal policy for calculated value function V):')
    grid_example.print_policy(optimal_policy)
    print('\nfinal V:')
    grid_example.print_values(V)
    print('\nfinal V:', V)

    print('\nstate visit counts:')
    print_state_visit_counts(grid_example, Q_counts)

    plt.plot(history)
    plt.show()
