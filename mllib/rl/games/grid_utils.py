import numpy as np

from rl.games.gridworld_base import gridworld_base
from rl.games.play_grid import play_episode_by_uniformly_random_actions


def gather_grid_state_samples(game, n_episodes=10000):
    all_states = []
    for _ in range(n_episodes):
        game.reset()
        _, _, episode_states = play_episode_by_uniformly_random_actions(game, on_invalid_action='no_effect')
        all_states.extend(episode_states)  # flatten all states
    return all_states


def grid_gather_state_action_samples(game, n_episodes=10000):
    all_states = []
    all_actions = []
    for _ in range(n_episodes):
        game.reset()
        episode_actions, _, episode_states = play_episode_by_uniformly_random_actions(game, on_invalid_action='no_effect')
        all_actions.extend(episode_actions[1:])  # flatten as one sample list
        all_states.extend(episode_states[:-1])  # flatten as one sample list
    return all_states, all_actions


class grid_state_action_encoder_decoder:
    def __init__(self):
        self.D = len(gridworld_base.ACTION_SPACE)
        self.action2id = {}
        self.id2action = {}
        for id, action in enumerate(gridworld_base.ACTION_SPACE):
            self.action2id[action] = id
            self.id2action[id] = action

    def action2vec(self, action):
        vec = np.zeros(self.D, dtype=int)
        vec[self.action2id[action]] = 1
        return vec

    def vec2action(self, vec):
        npvec = np.array(vec)
        assert npvec.shape == self.D
        one_idx = npvec.nonzero()
        assert len(one_idx) == 1
        return self.id2action[one_idx]

    def actions2vecs(self, actions):
        vecs = []
        for a in actions:
            vecs.append(self.action2vec(a))
        return vecs

    def vecs2actions(self, vecs):
        actions = []
        for v in vecs:
            actions.append(self.vec2action(v))
        return actions

    def encode_state_action(self, state, action):
        s = np.array(state, dtype=int)
        a = self.action2vec(action)
        sa = np.concatenate([s, a])
        return sa

    def dencode_state_action(self, state_action):
        s = state_action[:2]
        a = self.vec2action(state_action[2:])
        return s, a

    def encode_states_actions(self, states, actions):
        sas = []
        for s, a in zip(states, actions):
            sas.append(self.encode_state_action(s, a))
        return sas

    def dencode_state_action(self, states_actions):
        states = []
        actions = []
        for sa in states_actions:
            s, a = self.dencode_state_action(sa)
            states.append(s)
            actions.append(a)
        return states, actions
