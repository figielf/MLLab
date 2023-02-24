import matplotlib.pyplot as plt

from rl.temporal_difference.value_evaluation_by_td0 import temporal_difference_value_evaluation
from tests.rl.dynamic_programming.gridworld_examples import build_standart_simple_grid, build_negative_simple_grid, \
    build_windy_grid, grid_5x5, build_windy_grid_no_wind, build_windy_grid_penalized


def run_temporal_difference_value_evaluation(grid_factory, policy):
    print('grid rewards:')
    grid_examnple = grid_factory()
    grid_examnple.print_values(grid_examnple.rewards)
    print('policy:')
    grid_examnple.print_policy(policy)

    V, deltas = temporal_difference_value_evaluation(grid_factory, policy, n_episodes=10000)
    print('\nEvaluated Value:')
    grid_examnple.print_values(V)
    print('\nV:', V)

    plt.plot(deltas)
    plt.title('Q convergence')
    plt.show()
    return V


if __name__ == '__main__':
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    #run_temporal_difference_value_evaluation(grid_factory=lambda: build_standart_simple_grid(), policy=policy)
    run_temporal_difference_value_evaluation(grid_factory=lambda: build_negative_simple_grid(), policy=policy)
    #run_temporal_difference_value_evaluation(grid_factory=lambda: grid_5x5(), policy=policy)

    #run_temporal_difference_value_evaluation(grid_factory=lambda: build_windy_grid(quiet=True), policy=policy)
    #run_temporal_difference_value_evaluation(grid_factory=lambda: build_windy_grid_no_wind(quiet=True), policy=policy)
    #run_temporal_difference_value_evaluation(grid_factory=lambda: build_windy_grid_penalized(-0.2, quiet=True), policy=policy)
