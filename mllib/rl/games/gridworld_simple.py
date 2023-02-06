import numbers

from rl.games.gridworld_base import gridworld_base


def build_standart_simple_grid():
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    #
    # x means you can't go there
    # s means start position
    # number means reward at that state
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
        }
    return gridworld_simple(rows=3, columns=4, start=(2, 0), rewards=rewards, actions=actions)


def build_negative_simple_grid(step_cost=-0.1):
    # in this game will penalize every move so we want to try to minimize the number of moves
    g = build_standart_simple_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
        })
    return g


def grid_5x5(step_cost=-0.1):
    rewards = {(0, 4): 1, (1, 4): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'R'),
        (0, 3): ('L', 'D', 'R'),
        (1, 0): ('U', 'D', 'R'),
        (1, 1): ('U', 'D', 'L'),
        (1, 3): ('U', 'D', 'R'),
        (2, 0): ('U', 'D', 'R'),
        (2, 1): ('U', 'L', 'R'),
        (2, 2): ('L', 'R', 'D'),
        (2, 3): ('L', 'R', 'U'),
        (2, 4): ('L', 'U', 'D'),
        (3, 0): ('U', 'D'),
        (3, 2): ('U', 'D'),
        (3, 4): ('U', 'D'),
        (4, 0): ('U', 'R'),
        (4, 1): ('L', 'R'),
        (4, 2): ('L', 'R', 'U'),
        (4, 3): ('L', 'R'),
        (4, 4): ('L', 'U'),
        }
    visitable_state_rewards = {s: step_cost for s, _ in actions.items()}
    rewards.update(visitable_state_rewards)
    return gridworld_simple(rows=5, columns=5, start=(4, 0), rewards=rewards, actions=actions)


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

    def all_states(self):
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