def play_episode_by_deterministic_policy(game, policy, initial_action=None, max_steps=20, on_invalid_action='break'):
    assert on_invalid_action in ['break', 'no_effect']
    s = game.get_current_state()
    # states[t], reward[t] contains env state (response) on action[t]
    states = [s]
    actions = ['<START>']  # the <START> token indicates no action happened before start state
    rewords = [0]  # reward for initial state is set 0 (has now effect on V) as this is start environment state


    for i in range(max_steps):
        if game.is_game_over() or s not in policy:
            break

        if i == 0 and initial_action:
            a = initial_action
        else:
            a = policy[s]

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
