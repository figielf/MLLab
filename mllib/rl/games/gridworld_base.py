class gridworld_base:
    # gridworld base class with probabilistic rewards
    # assumed independent state transition and rewards distributions,
    # ie. p(next_state, reward|state, action) = p(reward|state, action) * p(next_state|state, action)
    ACTION_SPACE = ('U', 'D', 'L', 'R')

    def __init__(
            self,
            rows: int,
            columns: int,
            start: tuple[int, int],
            actions: dict[tuple[int, int], list[str]] = None):
        assert 0 <= start[0] < rows and 0 <= start[1] < columns
        self.n_rows = rows
        self.n_cols = columns
        self.game_start_state = start
        self.current_state = start
        self.actions = actions  # dict[state, list of possible actions from this state]

    def set_state(self, state):
        self.current_state = state

    def set_actions(self, actions):
        self.actions = actions

    def reset(self):
        self.current_state = self.game_start_state

    def get_current_state(self):
        return self.current_state

    def is_state_terminal(self, state):
        return len(self.get_valid_actions(state)) == 0  # there is no possible new state defined to move to by any of the actions

    def is_game_over(self):
        return self.is_state_terminal(self.current_state)

    def _move_impl(self, state, action) -> object:
        # should return new state after action from the state
        pass

    def try_move(self, state, action):
        assert self.is_valid_action(action, state)
        if action in self.actions[state]:
            return self._move_impl(state, action)
        else:
            print(f'The move: {action} from state: {state} is illegal')
            return None

    def move(self, action):
        assert self.is_valid_action(action)
        if self. is_game_over():
            print(f'The game is over!')
            return 0

        self.current_state = self._move_impl(self.current_state, action)
        return self.rewards.get(self.current_state, 0)

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.current_state
        return self.actions.get(state, [])

    def is_valid_action(self, action, state=None):
        if state is None:
            state = self.current_state
        return action in self.actions.get(state, [])

    def print_values(self, V):
        for i in range(self.n_rows):
            print("---------------------------")
            for j in range(self.n_cols):
                v = V.get((i, j), 0)
                if v >= 0:
                    print(" %.2f|" % v, end="")
                else:
                    print("%.2f|" % v, end="")
            print("")

    def print_policy(self, P):
        for i in range(self.n_rows):
            print("---------------------------")
            for j in range(self.n_cols):
                a = P.get((i, j), ' ')
                print("  %s  |" % a, end="")
            print("")

    def get_state_transition_probs(self):
        pass

    def get_reward_probs(self):
        pass
