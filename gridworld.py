'''
    This is a simple implementation of the two-agent Gridworld Cliff
    reinforcement learning task.

    Adapted from Example 6.6 (page 145) at:
    http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf

    The board is a 4x12 matrix, with (using Numpy matrix indexing):
        [0, 0] as the start at top-left
        [0, 11] as the goal at top-right
        [0, 1..10] as the cliff at top-center

    There are two agents that control one player, the first controlling the
    y-movement and the second controlling the x-movement. Each time step incurs
    -1 reward, and stepping into the cliff incurs -100 reward and a reset to
    the start. An episode terminates when the player reaches the goal.

    The action space indices correspond to:
    Agent 1: [up, stay, down]
    Agent 2: [left, stay, right]
'''

import numpy as np

num_agents = 2
grid_y, grid_x = 4, 12
start, goal = np.array((0, 0)), np.array((0, grid_x-1))

state_space = [np.array((i, j)) for i in range(grid_y) for j in range(grid_x)]
action_space = [np.array((i, j)) for i in (-1, 0, 1) for j in (-1, 0, 1)]

def perform_action(s, a_indices):
    '''Performs an action given by a_indices in state s. Returns:
       (s_next, reward)'''
    # Do some input validation
    a = a_indices_to_coordinates(a_indices)
    # assert included(s, state_space)
    # assert included(a, action_space)

    # Calculate the next state and reward
    sa = s + a
    if sa[0] == 0 and 1 <= sa[1] < (grid_x - 1):
        # The action moved us into the cliff, which resets the player to start
        s_next = start
        reward = -100
    else:
        s_next = sa
        reward = -1

    return (s_next, reward)

def perform_joint_action(s, joint_a):
    '''Performs an action given by joint_a in state s. Returns:
       (s_next, reward)'''
    a_indices = joint_action_to_indices(joint_a)
    return perform_action(s, a_indices)

def filter_actions(state, agent_no):
    '''Filter the actions available for an agent in a given state. Returns a
       bitmask of available actions.
       E.g. an agent in a corner is not allowed to move into a wall.'''
    avail_a = [1, 1, 1]
    # Vertical agent
    if agent_no == 0:
        if state[0] == 0: avail_a[0] = 0
        elif state[0] == grid_y-1: avail_a[2] = 0
    # Horizontal agent
    elif agent_no == 1:
        if state[1] == 0: avail_a[0] = 0
        elif state[1] == grid_x-1: avail_a[2] = 0
    return avail_a

def filter_joint_actions(state):
    '''Filters the joint actions available in a given state. Returns a bitmap
       of available actions.
       E.g. an agent in a corner is not allowed to move into a wall.'''
    avail_a = [1] * 9
    for i in range(len(action_space)):
        # Check if action moves us off the grid
        a = action_space[i]
        sa = state + a
        if (sa[0] < 0 or sa[0] >= grid_y) or (sa[1] < 0 or sa[1] >= grid_x):
            avail_a[i] = 0
    return avail_a

def start_state():
    '''Returns the start state of the game.'''
    return start

def is_end(s):
    '''Given a state, return if the game should end.'''
    return np.array_equal(s, goal)

def a_indices_to_coordinates(a_indices):
    '''Converts a list of action indices to action coordinates.'''
    coords = [i-1 for i in a_indices]
    return coords

def joint_action_to_indices(joint_a):
    '''Convert a joint action into action indices.'''
    a = action_space[joint_a] + 1
    return a.tolist()

def set_options(options):
    '''Set some game options, if given.'''
    global grid_x, grid_y, goal
    grid_x = options.get('grid_x', grid_x)
    grid_y = options.get('grid_y', grid_y)
    goal = np.array((0, grid_x-1))
    print(options)

def included(a, L):
    '''Returns if the np.array a is in a list of np.array's L.'''
    return any((l == a).all() for l in L)
