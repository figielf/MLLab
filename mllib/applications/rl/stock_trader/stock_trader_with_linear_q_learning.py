import itertools
import os
import pickle
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from applications.rl.stock_trader.action_value_model import sgd_with_momentum_linear_model
from applications.rl.stock_trader.agent import q_learning_agent
from applications.rl.stock_trader.environment_runner import play_episode, gather_stock_market_random_states_sample
from applications.rl.stock_trader.environment import stock_market
from tests.utils.timeseries_data_utils import get_stock_market_timeseries_data


def make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def run_agent(agent, env, scaler, training_mode, n_episodes=10000):
    history = []
    for i in range(n_episodes):
        final_portfolio_value, _ = play_episode(agent, env, scaler, in_train_mode=training_mode)
        history.append(final_portfolio_value)

        if (i + 1) % 100 == 0:
            print(f'Episode: {i + 1}/{n_episodes}, Reward: {final_portfolio_value}')
    return agent, history


def test_env(env, random_state=123):
    random.seed(random_state)
    single_stock_action_space = ['S', 'H', 'B']
    action_space_map = [list(i) for i in itertools.product(single_stock_action_space, repeat=3)]

    for i in range(100):
        old_state = env.current_state()
        old_value = env.calc_current_portfolio_value()
        a = random.randint(0, 26)
        # action = action_space_map[a]
        env.step(a)
        print(
            f'step: {i}, action: {a}, old: {old_value}, {old_state} ||| new: {env.current_state()}, {env.calc_current_portfolio_value()}')


if __name__ == '__main__':
    MODELS_FOLDER = 'saved_models'
    REWARDS_FOLDER = 'saved_rewards'
    N_EPISODES = 2000
    #BATCH_SIZE = 32
    INITIAL_CASH = 20000
    MODE = 'Train'
    #MODE = 'Test'

    make_dir(MODELS_FOLDER)
    make_dir(REWARDS_FOLDER)

    train_data, test_data = get_stock_market_timeseries_data(train_size=0.5)

    n_stocks = train_data.shape[1]
    state_space_dim = n_stocks * 2 + 1
    action_space_dim = n_stocks ** 3

    model_factory = lambda: sgd_with_momentum_linear_model(state_space_dim, action_space_dim)
    agent = q_learning_agent(state_space_dim, action_space_dim, model_factory)

    if MODE == 'Test':
        env = stock_market(INITIAL_CASH, test_data)
        #agent.epsilon = 0.01

        agent.load(f'{MODELS_FOLDER}/linear.npz')

        with open(f'{MODELS_FOLDER}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    elif MODE == 'Train':
        env = stock_market(INITIAL_CASH, train_data)

        scaler = StandardScaler()
        states_sample = gather_stock_market_random_states_sample(env)
        scaler.fit(states_sample)
    else:
        raise Exception(f'Only "Train" or "Test" MODE are supported')

    # test_env(env, random_state=98765)
    # env.reset()
    # test_env(env, random_state=12345)

    print(f'\n\n----------RUNNING {N_EPISODES} EPISODES----------')
    optimal_agent, portfolio_value_history = run_agent(agent, env, scaler, n_episodes=N_EPISODES, training_mode=(True if MODE == 'Train' else False))
    np.save(f'{REWARDS_FOLDER}/{MODE}.npy', portfolio_value_history)
    print('\n\n----------RUNNING EPISODES IS OVER----------')
    print(f'average final portfolio value: {np.mean(portfolio_value_history)}, mininum: {np.min(portfolio_value_history)}, maximum: {np.max(portfolio_value_history)}')

    if MODE == 'Train':
        agent.save(f'{MODELS_FOLDER}/linear.npz')

        with open(f'{MODELS_FOLDER}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        plt.plot(portfolio_value_history)
        plt.title('final value per episode')

        plt.subplot(1, 3, 2)
        plt.hist(portfolio_value_history, bins=20)
        plt.title('final value per episode histogram')

        plt.subplot(1, 3, 3)
        plt.plot(optimal_agent.model.history['mse'])
        plt.title('batch mse over all training history')
    else:
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(portfolio_value_history)
        plt.title('final value per episode')

        plt.subplot(1, 2, 2)
        plt.hist(portfolio_value_history, bins=20)
        plt.title('final value per episode histogram')
    plt.show()
