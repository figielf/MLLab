import numpy as np

from rl.games.play_grid import play_episode_by_deterministic_policy


def monte_carlo_value_evaluation(game_factory, policy, n_episodes=200, max_steps=20, gamma=0.9, mode='first_visit'):
    assert mode in ['first_visit', 'every_visit']  # [first visit MC, every visit MC]
    G_history = {}
    V = {}

    game_example = game_factory()
    for s in game_example.all_states():
        G_history[s] = []
        V[s] = 0

    for i in range(n_episodes):
        game = game_factory()
        game_all_states = list(game.all_states())
        start_state = game_all_states[np.random.randint(len(game_all_states))]
        game.set_state(start_state)  # set random start state
        _, rewords, states = play_episode_by_deterministic_policy(game, policy, max_steps=max_steps)

        G = 0
        T = len(rewords)
        for t in reversed(range(T - 1)):  # t = T-2, T-3, ..., 0
            state = states[t]
            reward_in_next_state = rewords[t + 1]
            G = reward_in_next_state + gamma * G

            if mode == 'first_visit':
                if state not in states[:t]:  # if state was not visited earlier (only visit is taken under consideration)
                    G_history[state].append(G)
                    V[state] = np.mean(G_history[state])
            elif mode == 'every_visit':
                G_history[state].append(G)
                V[state] = np.mean(G_history[state])
            else:
                raise Exception(f'Invalid mode: {mode} of MC visiting strategy. Supported modes are: "first_visit" and "every_visit"')
    return V
