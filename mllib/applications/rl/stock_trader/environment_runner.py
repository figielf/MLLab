import numpy as np


def play_episode(agent, env, scaler, in_train_mode=False):
    state = env.reset()
    state = scaler.transform([state])
    done = False
    mse_history = []
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])

        if in_train_mode:
            step_mse = agent.train(state, action, reward, next_state, done)
            mse_history.append(step_mse)

        state = next_state
    return info['current_portfolio_value'], mse_history


def gather_stock_market_random_states_sample(env):
    state = env.reset()
    states = []
    for _ in range(env.n_steps):
        action = np.random.choice(env.action_space)
        new_state, reward, done, info = env.step(action)
        state = new_state
        states.append(state)

        if done:
            break
    return states