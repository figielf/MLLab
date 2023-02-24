import numpy as np

from rl.games.grid_policies import generate_random_grid_policy
from rl.monte_carlo.play_grid import play_episode_by_deterministic_policy


def mean_count_quick(old_mean, n, x):
    new_n = n + 1
    lr = 1 / new_n
    new_mean = old_mean + lr * (x - old_mean)
    return new_mean, new_n


def monte_carlo_exploring_starts_deterministic_policy_optimization(game_factory, initial_policy=None, n_episodes=10000, max_steps=20, gamma=0.9, mode='first_visit'):
    assert mode in ['first_visit', 'every_visit']  # [first visit MC, every visit MC]

    Q_counts = {}
    Q = {}  # dict[state, action] = p(s', r | s, a)
    game_example = game_factory()
    for st, actions in game_example.actions.items():
        Q[st] = {}
        for ac in game_example.ACTION_SPACE:
            Q_counts[(st, ac)] = 0
            Q[st][ac] = 0

    if not initial_policy:
        initial_policy = generate_random_grid_policy(game_example)

    history = []
    policy = initial_policy
    for i in range(n_episodes):
        if i % 1000 == 0:
            print(f'episode {i}/{n_episodes}')

        biggest_change = 0
        game = game_factory()
        game_all_states = list(game.all_states())
        start_state = game_all_states[np.random.randint(len(game_all_states))]
        game.set_state(start_state)

        first_random_action = np.random.choice(game.ACTION_SPACE)
        actions, rewords, states = play_episode_by_deterministic_policy(game, policy, initial_action=first_random_action, max_steps=max_steps, on_invalid_action='no_effect')

        states_actions_pairs = list(zip(states[:-1], actions[1:]))
        # calculate Q
        G = 0
        T = len(rewords)
        for t in reversed(range(T - 1)):  # t = T-2, T-3, ..., 0
            state = states[t]
            action = actions[t + 1]
            reward_in_next_state = rewords[t + 1]
            G = reward_in_next_state + gamma * G

            old_q = Q[state][action]
            if mode == 'first_visit':
                if (state, action) not in states_actions_pairs[:t]:  # if (state, action) was not seen earlier (only visit is taken under consideration)
                    Q[state][action], Q_counts[(state, action)] = mean_count_quick(Q[state][action], Q_counts[(state, action)], G)
            elif mode == 'every_visit':
                Q[state][action], Q_counts[(state, action)] = mean_count_quick(Q[state][action], Q_counts[(state, action)], G)
            else:
                raise Exception(f'Invalid mode: {mode} of MC visiting strategy. Supported modes are: "first_visit" and "every_visit"')

            # choose best actions and improve policy
            policy[state], _ = get_best_action_and_q(Q[state])

            # calculate change in Q values
            biggest_change = max(biggest_change, np.abs(old_q - Q[state][action]))  # check convergence
        history.append(biggest_change)

    V = {state: get_best_action_and_q(actions_q)[1] for state, actions_q in Q.items()}
    return policy, V, Q, history
