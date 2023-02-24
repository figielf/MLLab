from matplotlib import pyplot as plt

from rl.temporal_difference.q_learning_policy_optimization import temporal_difference_q_learning_policy_evaluation
from tests.rl.dynamic_programming.gridworld_examples import grid_5x5, build_windy_grid_penalized, \
    build_standart_simple_grid, build_negative_simple_grid, build_windy_grid, build_windy_grid_no_wind
from tests.rl.monte_carlo.exploring_starts_policy_optimization_tests import print_state_visit_counts

if __name__ == '__main__':
    #grid_factory=lambda: build_standart_simple_grid()
    grid_factory=lambda: build_negative_simple_grid(step_cost=-0.1)
    #grid_factory=lambda: grid_5x5()

    #grid_factory=lambda: build_windy_grid(quiet=True)
    #grid_factory=lambda: build_windy_grid_no_wind(quiet=True)
    #grid_factory=lambda: build_windy_grid_penalized(-0.2, quiet=True)

    grid_example = grid_factory()
    print('\ngrid rewards:')
    grid_example.print_values(grid_example.rewards)

    optimal_policy, V, Q, history, Q_counts = temporal_difference_q_learning_policy_evaluation(game_factory=grid_factory, gamma=0.9, n_episodes=10000)
    print('\n\n----------OPTIMIZATION OVER----------')
    print('\nFinal policy (optimal policy for calculated value function V):')
    grid_example.print_policy(optimal_policy)
    print('\nfinal V:')
    grid_example.print_values(V)
    print('\nfinal V:', V)

    print('\nstate visit count rates:')
    print_state_visit_counts(grid_example, Q_counts, norm_counts=True)

    deltas, episode_reward = history
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(deltas)
    plt.title('Q convergence')
    plt.subplot(1, 2, 2)
    plt.plot(episode_reward)
    plt.title('per episode rewards')
    plt.show()
