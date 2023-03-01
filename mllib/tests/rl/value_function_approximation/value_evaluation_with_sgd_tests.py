import matplotlib.pyplot as plt

from rl.value_function_approximation.value_evaluation_with_sgd import \
    sgd_approximation_temporal_difference_value_evaluation, sgd_approximation_monte_carlo_value_evaluation
from tests.rl.dynamic_programming.gridworld_examples import build_standart_simple_grid, build_negative_simple_grid, \
    build_windy_grid, grid_5x5, build_windy_grid_no_wind, build_windy_grid_penalized


def run_sgd_approximation_value_evaluation(grid_factory, policy):
    print('grid rewards:')
    grid_examnple = grid_factory()
    grid_examnple.print_values(grid_examnple.rewards)
    print('policy:')
    grid_examnple.print_policy(policy)

    print('\n\ntemporal difference target approximation')
    V_td, episode_mse_td = sgd_approximation_temporal_difference_value_evaluation(grid_factory, policy, n_episodes=10000)
    print('\nEvaluated Value for temporal difference:')
    grid_examnple.print_values(V_td)
    print('\nV for temporal difference:', V_td)

    print('\n\nMonte Carlo target approximation')
    V_mc, episode_mse_mc = sgd_approximation_monte_carlo_value_evaluation(grid_factory, policy, n_episodes=10000)
    print('\nEvaluated Value for Monte Carlo:')
    grid_examnple.print_values(V_mc)
    print('\nV for Monte Carlo:', V_mc)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(episode_mse_td)
    plt.title('temporal difference episode MSE')

    plt.subplot(1, 2, 2)
    plt.plot(episode_mse_mc)
    plt.title('Monte Carlo episode MSE for Monte Carlo')
    plt.show()


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

    run_sgd_approximation_value_evaluation(grid_factory=lambda: build_standart_simple_grid(), policy=policy)
    #run_sgd_approximation_value_evaluation(grid_factory=lambda: build_negative_simple_grid(), policy=policy)
    #run_sgd_approximation_value_evaluation(grid_factory=lambda: grid_5x5(), policy=policy)

    #run_sgd_approximation_value_evaluation(grid_factory=lambda: build_windy_grid(quiet=True), policy=policy)
    #run_sgd_approximation_value_evaluation(grid_factory=lambda: build_windy_grid_no_wind(quiet=True), policy=policy)
    #run_sgd_approximation_value_evaluation(grid_factory=lambda: build_windy_grid_penalized(-0.2, quiet=True), policy=policy)
