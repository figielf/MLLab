from tests.rl.dynamic_programming.gridworld_examples import build_windy_grid_penalized
from rl.dynamic_programming.iterative_policy_evaluation import evaluate_policy


def run_policy_evaluation(grid, policy):
    grid.print_policy(policy)
    V = evaluate_policy(grid, policy, trace_logs=True)
    print("\nV:", V)
    return V


if __name__ == '__main__':

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L'}

    probabilistic_policy = {
        (2, 0): {'U': 0.5, 'R': 0.5},
        (1, 0): {'U': 1.0},
        (0, 0): {'R': 1.0},
        (0, 1): {'R': 1.0},
        (0, 2): {'R': 1.0},
        (1, 2): {'U': 1.0},
        (2, 1): {'R': 1.0},
        (2, 2): {'U': 1.0},
        (2, 3): {'L': 1.0},
    }

    odd_policy = {
        (0, 0): 'R',
        (0, 1): 'U',
        (0, 2): 'D',
        (1, 0): 'U',
        (1, 2): 'D',
        (2, 0): 'L',
        (2, 1): 'R',
        (2, 2): 'L',
        (2, 3): 'D'}


    #run_policy_evaluation(grid=build_standart_simple_grid(), policy=policy)
    #run_policy_evaluation(grid=build_negative_simple_grid(), policy=policy)
    #run_policy_evaluation(grid=grid_5x5(), policy=policy)

    #run_policy_evaluation(grid=build_windy_grid(), policy=probabilistic_policy)
    #run_policy_evaluation(grid=build_windy_grid_no_wind(), policy=probabilistic_policy)
    run_policy_evaluation(grid=build_windy_grid_penalized(), policy=probabilistic_policy)

    #run_policy_evaluation(grid=build_windy_grid(), policy=odd_policy)
