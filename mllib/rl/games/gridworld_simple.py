import numbers

from rl.games.gridworld_base import gridworld_base


class gridworld_simple(gridworld_base):
    # gridworld with deterministic state transition matrix

    def __init__(
            self,
            rows: int,
            columns: int,
            start: tuple[int, int],
            rewards: dict[tuple[int, int], float] = None,
            actions: dict[tuple[int, int], list[str]] = None):
        super().__init__(rows, columns, start, actions)
        self.rewards = rewards  # dict[state, reward]

    def set_rewards(self, rewards: dict[tuple[int, int], float]):
        self.rewards = rewards

    def all_states(self) -> set[tuple[int, int]]:
        return set(self.actions.keys()) | set(self.rewards.keys())

    def _move_impl(self, state: tuple[int, int], action: str) -> object:
        # should return new state after action from the state
        i, j = state
        if action == 'U':
            return i - 1, j
        if action == 'D':
            return i + 1, j
        if action == 'R':
            return i, j + 1
        if action == 'L':
            return i, j - 1

    def undo_move(self, state: tuple[int, int], action: str) -> tuple[int, int]:
        i, j = state
        if action == 'U':
            return i + 1, j
        if action == 'D':
            return i - 1, j
        if action == 'R':
            return i, j - 1
        if action == 'L':
            return i, j + 1

        old_state = self._undo_move_impl(self, self.current_state, action)
        if old_state in self.all_states():
            self.current_state = old_state
        else:
            print(f'The undo move: {action} from state: {self.current_state} is illegal, it leads to invalid state: {old_state}')

    def get_state_transition_probs(self) -> dict[tuple[tuple[int, int], str], dict[tuple[int, int], float]]:
        probs = {}
        for state, actions in self.actions.items():
            for a in actions:
                next_state = self._move_impl(state, a)
                probs[state, a] = {next_state: 1.0}
        return probs

    def get_reward_probs(self) -> dict[tuple[tuple[int, int], str, tuple[int, int]], dict[float, float]]:
        # returns returns distribution - dict[state, action, next_state] = dict[reward, probability]
        probs = {}
        for state in self.all_states():
            for action in self.actions.get(state, {}):
                if action in self.ACTION_SPACE:
                    next_state = self._move_impl(state, action)
                    reward = self.rewards.get(next_state, {0.0: 1.0})
                    if isinstance(reward, numbers.Number):
                        probs[state, action, next_state] = {reward: 1.0}
                        #probs[state, action, next_state] = {100.0: 0.0, reward + 0.8: 0.5, reward - 0.8: 0.5}
                    elif isinstance(reward, dict):
                        rewards_distrib = {}
                        for r, p in reward.items():
                            rewards_distrib[r] = p
                        probs[state, action, next_state] = rewards_distrib
                    else:
                        raise Exception(f'Unsupported type of rewards value. Expected number or dict, but was {reward} of type {type(reward)}')
        return probs