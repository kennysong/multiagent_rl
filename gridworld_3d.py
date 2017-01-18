'''
    This is a three-agent, 3-D version of the Gridworld Cliff reinforcement 
    learning task (see gridworld.py).

    The board is a 6 x 6 x 6 matrix, with (using Numpy matrix indexing):
        [3, 0] as the start at bottom-left
        [3, 11] as the goal at bottom-right
        [3, 1..10] as the cliff at bottom-center

    There are two agents that control one player, the first controlling the
    x-movement and the second controlling the y-movement. Each time step incurs
    -1 reward, and stepping into the cliff incurs -100 reward and a reset to
    the start. An episode terminates when the player reaches the goal.

    Note: the action space is {up, stay, down} x {left, stay, right},
    vectorized as {-1, 0, 1} x {-1, 0, 1}, including diagonal movements.
'''
