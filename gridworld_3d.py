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

import numpy as np
import gridworld

num_agents = 3
grid_z, grid_y, grid_x = 4, 4, 4
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

def perform_action(s, a_indices):
    '''Performs an action given by a_indices in state s. Returns:
       (s_next, reward)'''
    # Do some input validation
    a = a_indices_to_coordinates(a_indices)
    # assert included(s, state_space)
    # assert included(a, action_space)

    # Calculate the next state and reward
    reward = -1
    sa = s + a
    # The action moved us into the cliff, which resets the player to start
    if sa[0] == 0 and (
       (sa[1] == 0 and 1 <= sa[2] < grid_x) or
       (sa[1] == grid_y-1 and 0 <= sa[2] < grid_x-1) or
       (0 < sa[1] < grid_y-1)
    ):
        s_next = start
        reward = -100
    else:
        s_next = sa

    return (s_next, reward)

def perform_joint_action(s, joint_a):
    '''Performs an action given by joint_a in state s. Returns:
       (s_next, reward)'''
    a_indices = joint_action_to_indices(joint_a)
    return perform_action(s, a_indices)

def start_state():
    '''Returns the start state of the game.'''
    return start

def is_end(s):
    '''Given a state, return if the game should end.'''
    return np.array_equal(s, goal)

def filter_actions(state, agent_no):
    '''Filter the actions available for an agent in a given state. Returns a
       bitmap of available states.
       E.g. an agent in a corner is not allowed to move into a wall.'''
    actions = [1, 1, 1]
    # z-agent
    if agent_no == 0:
        if state[0] == 0: actions[0] = 0
        elif state[0] == grid_z-1: actions[2] = 0
    # y-agent
    elif agent_no == 1:
        if state[1] == 0: actions[0] = 0
        elif state[1] == grid_y-1: actions[2] = 0
    # x-agent
    elif agent_no == 2:
        if state[2] == 0: actions[0] = 0
        elif state[2] == grid_x-1: actions[2] = 0
    return actions

def filter_joint_actions(state):
    '''Filters the joint actions available in a given state. Returns a bitmap
       of available states.
       E.g. an agent in a corner is not allowed to move into a wall.'''
    avail_a = [1] * 27
    for i in range(len(action_space)):
        # Check if action moves us off the grid
        a = action_space[i]
        sa = state + a
        if (sa[0] < 0 or sa[0] >= grid_z) or \
           (sa[1] < 0 or sa[1] >= grid_y) or \
           (sa[2] < 0 or sa[2] >= grid_x):
            avail_a[i] = 0
    return avail_a

def a_indices_to_coordinates(a_indices):
    '''Converts a list of action indices to action coordinates.'''
    coords = [i-1 for i in a_indices]
    return coords

def joint_action_to_indices(joint_a):
    '''Convert a joint action into action indices.'''
    a = action_space[joint_a.index(1)] + 1
    return a.tolist()

def set_options(options):
    '''Set some game options, if given.'''
    global grid_x, grid_y, grid_z, goal
    grid_x = options.get('grid_x', grid_x)
    grid_y = options.get('grid_y', grid_y)
    grid_z = options.get('grid_z', grid_z)
    goal = np.array((0, grid_y-1, grid_x-1))

def included(a, L):
    '''Returns if the np.array a is in a list of np.array's L.'''
    return any((l == a).all() for l in L)
