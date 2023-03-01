import numpy as np
from sklearn.kernel_approximation import RBFSampler, Nystroem

from rl.monte_carlo.play_grid import play_episode_by_uniformly_random_actions, play_one_move_by_deterministic_policy, \
    play_episode_by_deterministic_policy


class approximation_function_state_value_evaluation_model:
    def __init__(self, states_transformer):
        self.states_transformer = states_transformer
        self.W = None

    def fit_preprocessor(self, states):
        self.states_transformer.fit(states)
        n_dims = self.states_transformer.n_components
        self.W = np.zeros(n_dims)

    def _approximate(self, x):
        return x.dot(self.W)

    def predict_one_sample(self, state):
        state_features = self.states_transformer.transform([state])[0]
        return self._approximate(state_features)

    def gradient_one_sample(self, state):
        state_features = self.states_transformer.transform([state])[0]
        return state_features


def gather_state_samples(game, n_episodes=10000):
    all_states = []
    for _ in range(n_episodes):
        game.reset()
        _, _, episode_states = play_episode_by_uniformly_random_actions(game, on_invalid_action='no_effect')
        all_states.extend(episode_states)  # flatten all states
    return all_states


def sgd_approximation_temporal_difference_value_evaluation(game_factory, policy, n_episodes=10000, gamma=0.9, learning_rate=0.01):
    # approximate Value function with target from temporal difference method: y = r + V[s']
    game_example = game_factory()
    sample_play_results = gather_state_samples(game_example)

    states_featurizer = RBFSampler()
    #states_featurizer = Nystroem()
    value_model = approximation_function_state_value_evaluation_model(states_featurizer)
    value_model.fit_preprocessor(sample_play_results)

    history = []
    for i in range(n_episodes):
        if (i + 1) % 1000 == 0:
            print(f'episode {i + 1}/{n_episodes}')

        game = game_factory()
        state = game.get_current_state()
        v_hat = value_model.predict_one_sample(state)

        episode_mse = []
        while not game.is_game_over():
            action, reward, new_state = play_one_move_by_deterministic_policy(game, policy, on_invalid_action='no_effect', with_epsilon_greedy=True)

            # as target set Value function from temporal difference: y = r + V[s']
            if game.is_state_terminal(new_state):
                target = reward
            else:
                new_state_v_hat = value_model.predict_one_sample(new_state)
                target = reward + gamma * new_state_v_hat

            error = v_hat - target
            dL_ds = error * value_model.gradient_one_sample(state)

            value_model.W = value_model.W - learning_rate * dL_ds

            episode_mse.append(error ** 2)
            v_hat = new_state_v_hat
            state = game.get_current_state()
        history.append(np.mean(episode_mse))

    V = {}
    for s in game_example.all_states():
        if game_example.is_state_terminal(s):
            V[s] = 0
        else:
            V[s] = value_model.predict_one_sample(s)

    return V, history


def sgd_approximation_monte_carlo_value_evaluation(game_factory, policy, n_episodes=200, max_steps=20, gamma=0.9, mode='first_visit', learning_rate=0.01):
    # approximate Value function with target from Monte Carlo method: y = G
    assert mode in ['first_visit', 'every_visit']  # [first visit MC, every visit MC]

    game_example = game_factory()
    sample_play_results = gather_state_samples(game_example)

    states_featurizer = RBFSampler()
    #states_featurizer = Nystroem()
    value_model = approximation_function_state_value_evaluation_model(states_featurizer)
    value_model.fit_preprocessor(sample_play_results)

    history = []
    for i in range(n_episodes):
        if (i + 1) % 1000 == 0:
            print(f'episode {i + 1}/{n_episodes}')

        game = game_factory()
        game_all_states = list(game.all_states())
        start_state = game_all_states[np.random.randint(len(game_all_states))]
        game.set_state(start_state)  # set random start state
        _, rewords, states = play_episode_by_deterministic_policy(game, policy, max_steps=max_steps)

        G = 0
        T = len(rewords)
        episode_mse = []
        for t in reversed(range(T - 1)):  # t = T-2, T-3, ..., 0
            state = states[t]
            v_hat = value_model.predict_one_sample(state)
            reward = rewords[t + 1]
            G = reward + gamma * G

            target = G
            error = v_hat - target
            dL_ds = error * value_model.gradient_one_sample(state)

            if mode == 'first_visit':
                if state not in states[:t]:  # if state was not visited earlier (only visit is taken under consideration)
                    value_model.W = value_model.W - learning_rate * dL_ds
                    episode_mse.append(error ** 2)
            elif mode == 'every_visit':
                value_model.W = value_model.W - learning_rate * dL_ds
                episode_mse.append(error ** 2)
            else:
                raise Exception(f'Invalid mode: {mode} of MC visiting strategy. Supported modes are: "first_visit" and "every_visit"')

        if len(episode_mse) > 0:
            history.append(np.mean(episode_mse))
        else:
            history.append(0)

        V = {}
        for s in game_example.all_states():
            if game_example.is_state_terminal(s):
                V[s] = 0
            else:
                V[s] = value_model.predict_one_sample(s)

    return V, history
