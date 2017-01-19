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

    Note: the action space is {up, stay, down} x {left, stay, right},
    vectorized as {-1, 0, 1} x {-1, 0, 1}, including diagonal movements.
'''

import numpy as np

grid_y, grid_x = 4, 4 # It was taking ages
state_space = [np.array((i, j)) for i in range(grid_y) for j in range(grid_x)]
action_space = [np.array((i, j)) for i in (-1, 0, 1) for j in (-1, 0, 1)]

start, goal = np.array((0, 0)), np.array((0, grid_x-1))
cliff_states = [np.array((0, x)) for x in range(1, grid_x-1)]

def P(s, s_next, a):
    '''The transition probabilities. Returns:
       P[s_{t+1} = s_next | s_t = s, a_t = a]'''
    # Check if the action moves us into the cliff, which always resets the
    # player to start
    if included(s + a, cliff_states):
        return 1 if np.array_equal(s_next, start) else 0

    # Check if the action moves us off the grid, which always results in stay
    if not included(s + a, state_space):
        return 1 if np.array_equal(s_next, s) else 0

    # Check that the action actually gets us to s_next
    if not np.array_equal(s + a, s_next):
        return 0

    # This must've been a valid transition
    return 1

def R(s, a):
    '''The reward function. Returns:
       E[R_{t+1} | s_t = s, a_t = a]'''
    # Check if the action moves us into the cliff, which gives -100 reward
    if included(s + a, cliff_states):
        return -100

    # Otherwise, return a default -1 reward per time step
    return -1

def perform_action(s, a):
    '''Performs action a in state s. Returns:
       (s_next, reward)'''
    # Do some input validation
    assert included(s, state_space)
    assert included(a, action_space)

    # Sample the next state based on the transition probabilities from P
    transition_probs = [P(s, s_next, a) for s_next in state_space]
    index = np.random.choice(range(len(transition_probs)), p=transition_probs)
    s_next = state_space[index]

    # Calculate the reward we recieved
    reward = R(s, a)

    return (s_next, reward)

def included(a, L):
    '''Returns if the np.array a is in a list of np.array's L.'''
    return any((l == a).all() for l in L)
