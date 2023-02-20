import numbers
import numpy as np

from rl.games.gridworld_simple import gridworld_base


class gridworld_windy(gridworld_base):
    # gridworld with probabilistic state transition matrix

    def __init__(
            self,
            rows: int,
            columns: int,
            start: tuple[int, int],
            rewards: dict[tuple[int, int], float] = None,
            transition_probs: dict[tuple[tuple[int, int], str], dict[tuple[int, int], float]] = None,
            actions: dict[tuple[int, int], list[str]] = None):
        super().__init__(rows, columns, start, actions)
        self.rewards = rewards  # dict[state, reward]
        self.transition_probs = None
        self.set_transition_probs(transition_probs)  # dict[tuple[state, action], dict[state, transition_probability]]

    def set_rewards(self, rewards: dict[tuple[int, int], float]):
        self.rewards = rewards

    def set_transition_probs(self, transition_probs: dict[tuple[tuple[int, int], str], dict[tuple[int, int], float]]):
        probs = {}
        for state_action, state_distribution in transition_probs.items():
            state, action = state_action
            if action in self.actions.get(state, []):  # if action is allowed for the state in the game
                probs[state_action] = state_distribution
            else:
                print(f'Action {action} is not a allowed for state {state}. Allowed actions are: {self.actions.get(state, [])}')

        self.transition_probs = probs  # dict[tuple[state, action], dict[state, transition_probability]]

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

    def _move_impl(self, state: tuple[int, int], action: str) -> object:
        # should return new state after action from the state
        transition_probs = self.transition_probs[state]
        states = list(transition_probs.kys())
        state_probs = list(transition_probs.vales())
        random_step_id = np.random.chice(len(state_probs), p=state_probs)
        return states[random_step_id]

    def get_state_transition_probs(self) -> dict[tuple[tuple[int, int], str], dict[tuple[int, int], float]]:
        return self.transition_probs

    def get_reward_probs(self) -> dict[tuple[tuple[int, int], str, tuple[int, int]], dict[float, float]]:
        # returns returns distribution - dict[state, action, next_state] = dict[reward, probability]
        probs = {}
        for state in self.all_states():
            for action in self.ACTION_SPACE:
                for next_state, _ in self.transition_probs.get((state, action), {}).items():  # if transition prob == 0 we reward should also be 0
                    reward = self.rewards.get(next_state, {0.0: 1.0})
                    if isinstance(reward, numbers.Number):
                        probs[state, action, next_state] = {reward: 1.0}
                    elif isinstance(reward, dict):
                        rewards_distrib = {}
                        for r, p in reward.items():
                            rewards_distrib[r] = p
                        probs[state, action, next_state] = rewards_distrib
                    else:
                        raise Exception(f'Unsupported type of rewards value. Expected number or dict, but was {reward} of type {type(reward)}')
        return probs
