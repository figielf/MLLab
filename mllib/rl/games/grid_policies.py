import numpy as np


def generate_random_grid_policy(grid):
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(grid.ACTION_SPACE)
    return policy


def generate_random_grid_Valid_policy(grid):
    policy = {}
    for s, actions in grid.actions.items():
        policy[s] = np.random.choice(actions)
    return policy
