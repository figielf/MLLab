import numpy as np
from sklearn.kernel_approximation import RBFSampler

from rl.games.grid_policies import generate_random_grid_policy
from rl.games.grid_utils import grid_state_action_encoder_decoder, grid_gather_state_action_samples
from rl.games.play_grid import play_episode_by_deterministic_policy, play_one_move_by_optimal_action_based_on_q
from rl.games.epsilon_greedy import get_epsilon_greedy_optimal_action_from_q, get_best_action_and_q


class approximation_function_action_value_evaluation_model:
    def __init__(self, states_transformer, states_action_encoder_decoder=grid_state_action_encoder_decoder()):
        self.states_transformer = states_transformer
        self.W = None
        self.sa_coder = states_action_encoder_decoder

    def fit_preprocessor(self, states, actions):
        states_actions = self.sa_coder.encode_states_actions(states, actions)
        self.states_transformer.fit(states_actions)
        n_dims = self.states_transformer.n_components
        self.W = np.zeros(n_dims)

    def _approximate(self, x):
        return x.dot(self.W)

    def predict_one_sample(self, state, action):
        state_action = self.sa_coder.encode_state_action(state, action)
        state_features = self.states_transformer.transform([state_action])[0]
        return self._approximate(state_features)

    def predict_all_actions(self, state, actions):
        print(tuple(state))
        predictions = {state: {}}
        for action in actions:
            predictions[state][action] = self.predict_one_sample(state, action)
        return predictions

    def gradient_one_sample(self, state, action):
        state_action = self.sa_coder.encode_state_action(state, action)
        state_features = self.states_transformer.transform([state_action])[0]
        return state_features


def calc_v_q_optimal_policy(game, model):
    Q = {}
    for s in game.all_states():
        Q[s] = {}
        for a in game.ACTION_SPACE:
            if game.is_state_terminal(s):
                Q[s][a] = 0
            else:
                Q[s][a] = model.predict_one_sample(s, a)

    V = {}
    target_policy = {}
    for state, actions_q in Q.items():
        best_q_action, V[state] = get_best_action_and_q(actions_q)
        if not game.is_state_terminal(state):
            target_policy[state] = best_q_action

    return Q, V, target_policy


def sgd_approximation_sarsa_policy_evaluation(game_factory, n_episodes=10000, gamma=0.9, alpha=0.1, eps=0.1, learning_rate=0.1):
    game_example = game_factory()
    sample_play_states, sample_play_actions = grid_gather_state_action_samples(game_example)
    states_featurizer = RBFSampler()
    #states_featurizer = Nystroem()
    q_model = approximation_function_action_value_evaluation_model(states_featurizer)
    q_model.fit_preprocessor(sample_play_states, sample_play_actions)

    history = []
    Q_counts = {}
    for i in range(n_episodes):
        if i % 2000 == 0:
            print(f'episode {i}/{n_episodes}')

        game = game_factory()
        state = game.get_current_state()
        q_hat = q_model.predict_all_actions(state, game.ACTION_SPACE)

        episode_reward = 0
        while not game.is_game_over():
            action, reward, new_state = play_one_move_by_optimal_action_based_on_q(game, q_hat, on_invalid_action='no_effect', with_epsilon_greedy=True, eps=eps)
            Q_counts[(state, action)] = Q_counts.get((state, action), 0) + 1

            # as target set Value function from sarsa: y = r + gamma * Q[s', a']
            if game.is_state_terminal(new_state):
                target = reward
            else:
                new_state_q_hat = q_model.predict_all_actions(new_state, game.ACTION_SPACE)
                next_action = get_epsilon_greedy_optimal_action_from_q(game.ACTION_SPACE, new_state_q_hat, new_state, eps=eps)
                target = reward + gamma * new_state_q_hat[new_state][next_action]

            error = q_hat[state][action] - target
            dL_ds = alpha * error * q_model.gradient_one_sample(state, action)

            q_model.W = q_model.W - learning_rate * dL_ds

            q_hat = new_state_q_hat
            state = game.get_current_state()

            episode_reward += reward
        history.append(episode_reward)

    Q, V, target_policy = calc_v_q_optimal_policy(game_example, q_model)
    return target_policy, V, Q, history, Q_counts


def sgd_approximation_q_learning_policy_evaluation(game_factory, n_episodes=10000, gamma=0.9, eps=0.1, learning_rate=0.1):
    game_example = game_factory()
    sample_play_states, sample_play_actions = grid_gather_state_action_samples(game_example)
    states_featurizer = RBFSampler()
    #states_featurizer = Nystroem()
    q_model = approximation_function_action_value_evaluation_model(states_featurizer)
    q_model.fit_preprocessor(sample_play_states, sample_play_actions)

    history = []
    Q_counts = {}
    for i in range(n_episodes):
        if i % 2000 == 0:
            print(f'episode {i}/{n_episodes}')

        game = game_factory()
        state = game.get_current_state()
        q_hat = q_model.predict_all_actions(state, game.ACTION_SPACE)

        episode_reward = 0
        while not game.is_game_over():
            action, reward, new_state = play_one_move_by_optimal_action_based_on_q(game, q_hat, on_invalid_action='no_effect', with_epsilon_greedy=True, eps=eps)
            Q_counts[(state, action)] = Q_counts.get((state, action), 0) + 1

            # as target set Value function from sarsa: y = r + gamma * max_a(Q[s', a'])
            if game.is_state_terminal(new_state):
                target = reward
            else:
                new_state_q_hat = q_model.predict_all_actions(new_state, game.ACTION_SPACE)
                _, max_q = get_best_action_and_q(new_state_q_hat[new_state])
                target = reward + gamma * max_q

            error = q_hat[state][action] - target
            dL_ds = error * q_model.gradient_one_sample(state, action)

            q_model.W = q_model.W - learning_rate * dL_ds

            q_hat = new_state_q_hat
            state = game.get_current_state()

            episode_reward += reward
        history.append(episode_reward)

    Q, V, target_policy = calc_v_q_optimal_policy(game_example, q_model)
    return target_policy, V, Q, history, Q_counts


def sgd_approximation_monte_carlo_epsilon_greedy_deterministic_policy_optimization(game_factory, initial_policy=None, n_episodes=20000, max_steps=20, gamma=0.9, learning_rate=0.1, mode='first_visit'):
    return sgd_approximation_monte_carlo_exploring_starts_deterministic_policy_optimization(game_factory, initial_policy, n_episodes, max_steps, gamma, learning_rate, mode, explore_mode='epsilon_greedy')


def sgd_approximation_monte_carlo_exploring_starts_deterministic_policy_optimization(game_factory, initial_policy=None, n_episodes=20000, max_steps=20, gamma=0.9, learning_rate=0.1, mode='first_visit', explore_mode='exploring_starts'):
    assert mode in ['first_visit', 'every_visit']  # [first visit MC, every visit MC]
    assert explore_mode in ['exploring_starts', 'epsilon_greedy']  # [exploring starts MC, epsilon greedy MC]

    game_example = game_factory()
    sample_play_states, sample_play_actions = grid_gather_state_action_samples(game_example)
    states_featurizer = RBFSampler()
    #states_featurizer = Nystroem()
    q_model = approximation_function_action_value_evaluation_model(states_featurizer)
    q_model.fit_preprocessor(sample_play_states, sample_play_actions)

    if not initial_policy:
        initial_policy = generate_random_grid_policy(game_example)

    history = []
    Q_counts = {}
    policy = initial_policy
    for i in range(n_episodes):
        if i % 2000 == 0:
            print(f'episode {i}/{n_episodes}')

        biggest_change = 0
        game = game_factory()
        if explore_mode == 'exploring_starts':
            # explore random starting state
            game_all_states = list(game.all_states())
            start_state = game_all_states[np.random.randint(len(game_all_states))]
            game.set_state(start_state)

            # choose random first action so that all (s, a) are explored to get data for Q
            first_action = np.random.choice(game.ACTION_SPACE)
            use_epsilon_greedy = False
        elif explore_mode == 'epsilon_greedy':
            # take action from policy and explore by epsilon greedy
            first_action = None
            use_epsilon_greedy = True
        else:
            raise Exception(f'Invalid  mode: {mode} of MC exploring strategy. Supported modes are: "exploring_starts" and "epsilon_greedy"')

        actions, rewords, states = play_episode_by_deterministic_policy(game, policy, initial_action=first_action, max_steps=max_steps, on_invalid_action='no_effect', with_epsilon_greedy=use_epsilon_greedy)

        states_actions_pairs = list(zip(states[:-1], actions[1:]))
        # calculate Q
        G = 0
        episode_reward = 0
        T = len(rewords)
        for t in reversed(range(T - 1)):  # t = T-2, T-3, ..., 0
            state = states[t]
            action = actions[t + 1]
            Q_counts[(state, action)] = Q_counts.get((state, action), 0) + 1
            q_hat = q_model.predict_all_actions(state, game.ACTION_SPACE)

            reward = rewords[t + 1]
            G = reward + gamma * G

            target = G
            error = q_hat[state][action] - target
            dL_ds = error * q_model.gradient_one_sample(state, action)

            if mode == 'first_visit':
                if (state, action) not in states_actions_pairs[:t]:  # if (state, action) was not seen earlier (only visit is taken under consideration)
                    q_model.W = q_model.W - learning_rate * dL_ds
            elif mode == 'every_visit':
                q_model.W = q_model.W - learning_rate * dL_ds
            else:
                raise Exception(f'Invalid mode: {mode} of MC visiting strategy. Supported modes are: "first_visit" and "every_visit"')

            # choose best actions and improve policy
            new_q_hat = q_model.predict_all_actions(state, game.ACTION_SPACE)
            policy[state], _ = get_best_action_and_q(new_q_hat[state])

            episode_reward += reward
        history.append(episode_reward)

    Q, V, target_policy = calc_v_q_optimal_policy(game_example, q_model)
    return target_policy, V, Q, history, Q_counts




