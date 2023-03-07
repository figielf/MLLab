import gym
import numpy as np


class cartpole_state_action_encoder:
    def encode_state_action(self, state, action):
        return np.concatenate((state, [action]))

    def encode_states_actions(self, states, actions):
        sas = []
        for s, a in zip(states, actions):
            sas.append(self.encode_state_action(s, a))
        return sas


def get_cartpole_action_space(env):
    return [i for i in range(env.action_space.n)]


def gather_cartpole_samples(game, n_episodes=10000):
    all_states = []
    all_actions = []
    for _ in range(n_episodes):
        state, info = game.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = game.action_space.sample()
            all_states.append(state)
            all_actions.append(action)

            state, reward, done, truncated, info = game.step(action)
    state, info = game.reset()
    return all_states, all_actions


def cartpole_factory():
    return gym.make("CartPole-v1", render_mode="rgb_array")


def predict_all_actions(model, state, action_space):
    return [model.predict_one_sample(state, a) for a in action_space]


def get_optimal_with_epsilon_greedy_action(action_space, model, state, eps):
    # explore by epsilon greedy strategy
    p = np.random.random()
    if p < eps:
        action = np.random.choice(action_space)
    else:
        q_hat = predict_all_actions(model, state, action_space)
        action = np.argmax(q_hat)
    return action


def run_cartpole_agent(model, env_factory, eps=0.0, n_episodes=20, render=False):
    env = env_factory()
    action_space = get_cartpole_action_space(env)
    reward_per_episode = []
    for _ in range(n_episodes):
        done = False
        truncated = False
        episode_reward = 0
        state, info = env.reset()
        while not (done or truncated):
            a = get_optimal_with_epsilon_greedy_action(action_space, model, state, eps=eps)
            state, reward, done, truncated, info = env.step(a)
            episode_reward += reward
            if render:
                env.render()
        reward_per_episode.append(episode_reward)
    return reward_per_episode

