'''
    This is a simple implementation of the two-agent Gridworld Cliff
    reinforcement learning task.

    Adapted from Example 6.6 at:
    https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html

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
    reward = -1
    sa = s + a
    # The action moved us into the cliff, which resets the player to start
    if sa[0] == 0 and 1 <= sa[1] < (grid_x - 1):
        s_next = start
        reward = -100
    # The action moved us off the grid, which results in stay
    elif (sa[0] < 0 or sa[0] >= grid_y) or (sa[1] < 0 or sa[1] >= grid_x):
        s_next = s
    else:
        s_next = sa

    return (s_next, reward)

def filter_action_space(dist, state, agent_no):
    '''Filter the actions available (dist holds probabilities of all actions)
       for an agent in a given state.
       E.g. an agent in a corner is not allowed to move into a wall.'''
    # dist = dist[0].data.clone().numpy()
    dist = np.copy(dist)
    # Vertical agent
    if agent_no == 0:
        if state[0] == 0: dist[0] = 0
        elif state[0] == grid_y-1: dist[2] = 0
    # Horizontal agent
    elif agent_no == 1:
        if state[1] == 0: dist[0] = 0
        elif state[1] == grid_x-1: dist[2] = 0
    # Renomalize probability distribution
    dist /= sum(dist)
    return dist

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

def set_options(options):
    '''Set some game options, if given.'''
    global grid_x, grid_y, goal
    grid_x = options.get('grid_x', grid_x)
    grid_y = options.get('grid_y', grid_y)
    goal = np.array((0, grid_x-1))

def included(a, L):
    '''Returns if the np.array a is in a list of np.array's L.'''
    return any((l == a).all() for l in L)
