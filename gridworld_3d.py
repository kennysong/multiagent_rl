'''
    This is a three-agent, 3-D version of the Gridworld Cliff reinforcement
    learning task (see gridworld.py).

    The board is a 6 x 6 x 6 matrix, with (using Numpy matrix indexing):
        [0, 0, 0] as the start at the corner
        [0, 0, 5] as the goal, diagonally across the start on the same level
        [0, 0..5, 0..5] as the cliff, except for the start and goal

    There are three agents that control one player, each controlling the z, y,
    and x directions. Each time step incurs -1 reward, and stepping into the
    cliff incurs -100 reward and a reset to the start. An episode terminates
    when the player reaches the goal.

    Note: the action space is
    {z--, stay, z++} x {y--, stay, y++} x {x--, stay, x++}
    = {-1, 0, 1} x {-1, 0, 1} x {-1, 0, 1}.
'''

# TODO: Make this conform to the policy_gradient game interface

import numpy as np
import gridworld
from gridworld import P, R, perform_action, included

grid_z, grid_y, grid_x = 6, 6, 6
state_space = [np.array((z, y, x)) for z in range(grid_z)
                                   for y in range(grid_y)
                                   for x in range(grid_x)]
action_space = [np.array((z, y, x)) for z in (-1, 0, 1)
                                    for y in (-1, 0, 1)
                                    for x in (-1, 0, 1)]

start, goal = np.array((0, 0, 0)), np.array((0, grid_y-1, grid_x-1))
cliff_states = [np.array((0, y, x)) for y in range(grid_y)
                                    for x in range(grid_x)
                                    if any(np.array((0, y, x)) != start) and
                                       any(np.array((0, y, x)) != goal)]

# Monkey patch global variables
gridworld.state_space = state_space
gridworld.action_space = action_space
gridworld.start = start
gridworld.goal = goal
gridworld.cliff_states = cliff_states
