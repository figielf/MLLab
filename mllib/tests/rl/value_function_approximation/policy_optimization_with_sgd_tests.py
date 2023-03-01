from matplotlib import pyplot as plt

from rl.value_function_approximation.policy_optimization_with_sgd import sgd_approximation_sarsa_policy_evaluation, \
    sgd_approximation_q_learning_policy_evaluation, \
    sgd_approximation_monte_carlo_epsilon_greedy_deterministic_policy_optimization, \
    sgd_approximation_monte_carlo_exploring_starts_deterministic_policy_optimization
from tests.rl.dynamic_programming.gridworld_examples import grid_5x5, build_windy_grid_penalized, \
    build_standart_simple_grid, build_negative_simple_grid, build_windy_grid, build_windy_grid_no_wind
from tests.rl.monte_carlo.exploring_starts_policy_optimization_tests import print_state_visit_counts

if __name__ == '__main__':
    N_EPISODES = 20000

    #grid_factory=lambda: build_standart_simple_grid()
    grid_factory=lambda: build_negative_simple_grid(step_cost=-0.1)
    #grid_factory=lambda: grid_5x5()

    #grid_factory=lambda: build_windy_grid(quiet=True)
    #grid_factory=lambda: build_windy_grid_no_wind(quiet=True)
    #grid_factory=lambda: build_windy_grid_penalized(-0.2, quiet=True)

    grid_example = grid_factory()
    print('\ngrid rewards:')
    grid_example.print_values(grid_example.rewards)

    print('\n\ntemporal difference epsilon greedy Monte Carlo target approximation')
    optimal_policy_mc_epsilon, V_mc_epsilon, Q_mc_epsilon, episode_reward_mc_epsilon, Q_counts_mc_epsilon = sgd_approximation_monte_carlo_epsilon_greedy_deterministic_policy_optimization(game_factory=grid_factory, gamma=0.9, n_episodes=N_EPISODES)
    print('\n\n----------EPSILON GREEDY MC OPTIMIZATION OVER----------')
    print('\nFinal policy (optimal policy for calculated value function V) for temporal difference epsilon greedy Monte Carlo:')
    grid_example.print_policy(optimal_policy_mc_epsilon)
    print('\nfinal V for temporal difference epsilon greedy Monte Carlo:')
    grid_example.print_values(V_mc_epsilon)
    print('\nfinal V for temporal difference epsilon greedy Monte Carlo:', V_mc_epsilon)
    print('\nstate visit count rates:')
    print_state_visit_counts(grid_example, Q_counts_mc_epsilon, norm_counts=False)

    print('\n\ntemporal difference exploring starts Monte Carlo target approximation')
    optimal_policy_mc_exploring, V_mc_exploring, Q_mc_exploring, episode_reward_mc_exploring, Q_counts_mc_exploring = sgd_approximation_monte_carlo_exploring_starts_deterministic_policy_optimization(game_factory=grid_factory, gamma=0.9, n_episodes=N_EPISODES)
    print('\n\n----------EXPLORING STARTS MC OPTIMIZATION OVER----------')
    print('\nFinal policy (optimal policy for calculated value function V) for temporal difference exploring starts Monte Carlo:')
    grid_example.print_policy(optimal_policy_mc_exploring)
    print('\nfinal V for temporal difference exploring starts Monte Carlo:')
    grid_example.print_values(V_mc_exploring)
    print('\nfinal V for temporal difference exploring starts Monte Carlo:', V_mc_exploring)
    print('\nstate visit count rates:')
    print_state_visit_counts(grid_example, Q_counts_mc_exploring, norm_counts=False)

    print('\n\ntemporal difference q-learning target approximation')
    optimal_policy_qlearning, V_qlearning, Q_qlearning, episode_reward_qlearning, Q_counts_qlearning = sgd_approximation_q_learning_policy_evaluation(game_factory=grid_factory, gamma=0.9, n_episodes=N_EPISODES)
    print('\n\n----------Q-LEARNING OPTIMIZATION OVER----------')
    print('\nFinal policy (optimal policy for calculated value function V) for temporal difference q-learning:')
    grid_example.print_policy(optimal_policy_qlearning)
    print('\nfinal V for temporal difference q-learning:')
    grid_example.print_values(V_qlearning)
    print('\nfinal V for temporal difference q-learning:', V_qlearning)
    print('\nstate visit count rates:')
    print_state_visit_counts(grid_example, Q_counts_qlearning, norm_counts=False)

    print('\n\ntemporal difference sarsa target approximation')
    optimal_policy_sarsa, V_sarsa, Q_sarsa, episode_reward_sarsa, Q_counts_sarsa = sgd_approximation_sarsa_policy_evaluation(game_factory=grid_factory, gamma=0.9, n_episodes=N_EPISODES)
    print('\n\n----------SARSA OPTIMIZATION OVER----------')
    print('\nFinal policy (optimal policy for calculated value function V) for temporal difference sarsa:')
    grid_example.print_policy(optimal_policy_sarsa)
    print('\nfinal V for temporal difference sarsa:')
    grid_example.print_values(V_sarsa)
    print('\nfinal V for temporal difference sarsa:', V_sarsa)
    print('\nstate visit count rates:')
    print_state_visit_counts(grid_example, Q_counts_sarsa, norm_counts=False)

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.plot(episode_reward_mc_epsilon)
    plt.title('per episode epsilon greedy Monte Carlo rewards')
    plt.subplot(2, 2, 2)
    plt.plot(episode_reward_mc_exploring)
    plt.title('per episode exploring starts Monte Carlo rewards')
    plt.subplot(2, 2, 3)
    plt.plot(episode_reward_qlearning)
    plt.title('per episode q-learning rewards')
    plt.subplot(2, 2, 4)
    plt.plot(episode_reward_sarsa)
    plt.title('per episode sarsa rewards')
    plt.show()
