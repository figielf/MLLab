from rl.dynamic_programming.value_iteration import calculate_optimized_value
from tests.rl.dynamic_programming.gridworld_examples import grid_5x5, build_windy_grid_penalized, \
    build_windy_grid_no_wind, build_windy_grid, build_negative_simple_grid, build_standart_simple_grid


def run_value_iteration_optimization(grid):
    V = calculate_optimized_value(grid, trace_logs=True)
    print("\nV:", V)
    return V


if __name__ == '__main__':
    #run_value_iteration_optimization(grid=build_standart_simple_grid())
    #run_value_iteration_optimization(grid=build_negative_simple_grid())
    #run_value_iteration_optimization(grid=grid_5x5())

    #run_value_iteration_optimization(grid=build_windy_grid())
    #run_value_iteration_optimization(grid=build_windy_grid_no_wind())
    run_value_iteration_optimization(grid=build_windy_grid_penalized(-0.2))
