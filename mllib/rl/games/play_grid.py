import numpy as np

from rl.games.epsilon_greedy import get_epsilon_greedy_policy_action, get_epsilon_greedy_optimal_action_from_q, \
    get_optimal_action_from_q


def play_one_move_by_optimal_action_based_on_q(game, Q, state=None, on_invalid_action='break', with_epsilon_greedy=False, eps=0.1):
    assert on_invalid_action in ['break', 'no_effect']

    if not state:
        state = game.get_current_state()

    if game.is_game_over() or state not in Q:
        return None

    if with_epsilon_greedy:
        action = get_epsilon_greedy_optimal_action_from_q(game.ACTION_SPACE, Q, state, eps)
    else:
        action = get_optimal_action_from_q(Q, state)

    if on_invalid_action == 'no_effect':
        if game.is_valid_action(action):  # count invalid action as a move but stay in the same position and return reward for this position
            reward = game.move(action)
        else:
            reward = game.rewards.get(state, 0)
    elif on_invalid_action == 'break':
        reward = game.move(action)  # in case of invalid action game will raise an error
    else:
        raise Exception(f'Invalid mode: {on_invalid_action} for on_invalid_action. Supported modes are: "break" and "no_effect"')

    new_state = game.get_current_state()
    return action, reward, new_state


def play_one_move_by_deterministic_policy(game, policy, state=None, on_invalid_action='break', with_epsilon_greedy=False, eps=0.1):
    assert on_invalid_action in ['break', 'no_effect']

    if not state:
        state = game.get_current_state()

    if game.is_game_over() or state not in policy:
        return None

    if with_epsilon_greedy:
        action = get_epsilon_greedy_policy_action(game.ACTION_SPACE, policy, state, eps)
    else:
        action = policy[state]

    if on_invalid_action == 'no_effect':
        if game.is_valid_action(action):  # count invalid action as a move but stay in the same position and return reward for this position
            reward = game.move(action)
        else:
            reward = game.rewards.get(state, 0)
    elif on_invalid_action == 'break':
        reward = game.move(action)  # in case of invalid action game will raise an error
    else:
        raise Exception(f'Invalid mode: {on_invalid_action} for on_invalid_action. Supported modes are: "break" and "no_effect"')

    new_state = game.get_current_state()
    return action, reward, new_state


def play_episode_by_deterministic_policy(game, policy, initial_action=None, max_steps=20, on_invalid_action='break', with_epsilon_greedy=False, eps=0.1):
    assert on_invalid_action in ['break', 'no_effect']
    s = game.get_current_state()
    # states[t], reward[t] contains env state (response) on action[t]
    states = [s]
    actions = ['<START>']  # the <START> token indicates no action happened before start state
    rewords = [0]  # reward for initial state is set 0 (has now effect on V) as this is start environment state


    for i in range(max_steps):
        if game.is_game_over():
            break

        if i == 0 and initial_action:
            a = initial_action
        else:
            if s in policy:
                a = policy[s]
            else:
                break

        # explore by epsilon greedy strategy
        if with_epsilon_greedy:
            p = np.random.random()
            if p < eps:
                a = np.random.choice(game.ACTION_SPACE)

        if on_invalid_action == 'no_effect':
            if game.is_valid_action(a):  # count invalid action as a move but stay in the same position and return reward for this position
                r = game.move(a)
            else:
                r = game.rewards.get(s, 0)
        elif on_invalid_action == 'break':
            r = game.move(a)  # in case of invalid action game will raise an error
        else:
            raise Exception(f'Invalid mode: {on_invalid_action} for on_invalid_action. Supported modes are: "break" and "no_effect"')

        s = game.get_current_state()

        actions.append(a)
        states.append(s)
        rewords.append(r)

    assert len(actions) == len(rewords)
    assert len(actions) == len(states)
    return actions, rewords, states


def play_episode_by_uniformly_random_actions(game, on_invalid_action='break'):
    assert on_invalid_action in ['break', 'no_effect']

    s = game.get_current_state()
    states = [s]
    actions = ['<START>']  # the <START> token indicates no action happened before start state
    rewords = [0]  # reward for initial state is set 0 (has now effect on V) as this is start environment state
    while not game.is_game_over():
        a = np.random.choice(game.ACTION_SPACE)

        if on_invalid_action == 'no_effect':
            if game.is_valid_action(a):  # count invalid action as a move but stay in the same position and return reward for this position
                r = game.move(a)
            else:
                r = game.rewards.get(s, 0)
        elif on_invalid_action == 'break':
            r = game.move(a)  # in case of invalid action game will raise an error
        else:
            raise Exception(f'Invalid mode: {on_invalid_action} for on_invalid_action. Supported modes are: "break" and "no_effect"')

        s = game.get_current_state()
        actions.append(a)
        states.append(s)
        rewords.append(r)

    assert len(actions) == len(rewords)
    assert len(actions) == len(states)
    return actions, rewords, states
