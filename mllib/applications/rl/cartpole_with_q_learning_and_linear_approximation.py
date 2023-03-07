import gym
import numpy as np
from matplotlib import pyplot as plt
from sklearn.kernel_approximation import RBFSampler

from rl.games.openai_gym.cartpole_utils import cartpole_state_action_encoder, gather_cartpole_samples, \
    predict_all_actions, get_optimal_with_epsilon_greedy_action, get_cartpole_action_space, run_cartpole_agent
from rl.value_function_approximation.policy_optimization_with_sgd import \
    approximation_function_action_value_evaluation_model


def sgd_approximation_q_learning_policy_evaluation(game_factory, n_episodes=10000, gamma=0.9, eps=0.1, learning_rate=0.1):
    game = game_factory()
    action_space = get_cartpole_action_space(game)
    sample_play_states, sample_play_actions = gather_cartpole_samples(game)
    states_featurizer = RBFSampler()
    #states_featurizer = Nystroem()
    q_model = approximation_function_action_value_evaluation_model(states_featurizer, states_action_encoder_decoder=cartpole_state_action_encoder())
    q_model.fit_preprocessor(sample_play_states, sample_play_actions)

    history = []
    for i in range(n_episodes):
        state, info = game.reset()

        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = get_optimal_with_epsilon_greedy_action(action_space, q_model, state, eps)
            new_state, reward, done, truncated, info = game.step(action)

            # as target set Value function from sarsa: y = r + gamma * max_a(Q[s', a'])
            if done:
                target = reward
            else:
                new_state_q_hat = predict_all_actions(q_model, new_state, action_space)
                target = reward + gamma * max(new_state_q_hat)

            error = q_model.predict_one_sample(state, action) - target
            dL_ds = error * q_model.gradient_one_sample(state, action)
            q_model.W = q_model.W - learning_rate * dL_ds

            episode_reward += reward
            state = new_state
        history.append(episode_reward)

        if (i + 1) % 50 == 0:
            print(f'Episode: {i + 1}/{n_episodes}, Reward: {episode_reward}')

        # early exit
        if i > 20 and np.mean(history[-20:]) == 500:
            print('Best possible reward achieve. Exit early.')
            break

    return history, q_model, game


if __name__ == '__main__':
    N_EPISODES = 1500

    cartpole_factory = lambda: gym.make('CartPole-v1', render_mode='rgb_array')

    print('\n\ntemporal difference q-learning target approximation')
    episode_reward_qlearning, model, model_env = sgd_approximation_q_learning_policy_evaluation(game_factory=cartpole_factory, gamma=0.9, n_episodes=N_EPISODES)
    print('\n\n----------Q-LEARNING OPTIMIZATION OVER----------')

    # test trained agent
    test_reward = run_cartpole_agent(model, lambda: model_env, eps=0.0, n_episodes=20)
    print(f'Average test reward: {np.mean(test_reward)}')

    # watch trained agent
    new_env = gym.make('CartPole-v1', render_mode='human')
    run_cartpole_agent(model, lambda: new_env, n_episodes=1, eps=0.0, render=True)

    plt.plot(episode_reward_qlearning)
    plt.title('reward per episode')
    plt.show()
