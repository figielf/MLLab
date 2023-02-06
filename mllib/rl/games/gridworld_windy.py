import numbers
import numpy as np

from rl.games.gridworld_simple import gridworld_base


def build_windy_grid():
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

    # p(s' | s, a) represented as:
    # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
    probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'D'): {(2, 0): 1.0},
        ((2, 0), 'L'): {(2, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((1, 0), 'L'): {(1, 0): 1.0},
        ((1, 0), 'R'): {(1, 0): 1.0},
        ((0, 0), 'U'): {(0, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'L'): {(0, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'U'): {(0, 1): 1.0},
        ((0, 1), 'D'): {(0, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'U'): {(0, 2): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((2, 1), 'U'): {(2, 1): 1.0},
        ((2, 1), 'D'): {(2, 1): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'D'): {(2, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'D'): {(2, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((2, 3), 'R'): {(2, 3): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'L'): {(1, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
        }
    return gridworld_windy(rows=3, columns=4, start=(2, 0), rewards=rewards, transition_probs=probs, actions=actions)


def build_windy_grid_no_wind():
    g = build_windy_grid()
    g.transition_probs[((1, 2), 'U')] = {(0, 2): 1.0}
    return g


def build_windy_grid_penalized(step_cost=-0.1):
    rewards = {
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
        (0, 3): 1,
        (1, 3): -1
        }
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

    # p(s' | s, a) represented as:
    # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
    probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'D'): {(2, 0): 1.0},
        ((2, 0), 'L'): {(2, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((1, 0), 'L'): {(1, 0): 1.0},
        ((1, 0), 'R'): {(1, 0): 1.0},
        ((0, 0), 'U'): {(0, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'L'): {(0, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'U'): {(0, 1): 1.0},
        ((0, 1), 'D'): {(0, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'U'): {(0, 2): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((2, 1), 'U'): {(2, 1): 1.0},
        ((2, 1), 'D'): {(2, 1): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'D'): {(2, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'D'): {(2, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((2, 3), 'R'): {(2, 3): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'L'): {(1, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
        }
    return gridworld_windy(rows=3, columns=4, start=(2, 0), rewards=rewards, transition_probs=probs, actions=actions)


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
        self.transition_probs = transition_probs  # dict[tuple[state, action], dict[state, transition_probability]]

    def set_rewards(self, rewards: dict[tuple[int, int], float]):
        self.rewards = rewards

    def set_tansition_probs(self, transition_probs: dict[tuple[tuple[int, int], str], dict[tuple[int, int], float]]):
        self.transition_probs = transition_probs

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
