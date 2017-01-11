'''
    Very basic testing for gridworld.py by selecting random actions and
    printing out the game state. You may want to decrease gridworld.grid_cols
    so the random agent can find the goal in a reasonable time.
'''


# Add parent directory to PYTHONPATH
# (http://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import gridworld as gw
import numpy as np
import random

if __name__ == '__main__':
    player = gw.start

    while True:
        board = np.zeros((gw.grid_rows, gw.grid_cols))
        board[player[0], player[1]] = 1

        if not np.array_equal(player, gw.goal):
            action = random.choice(gw.action_space)
            print('Player at: {} Random Action: {}'.format(player, action))
            print(board)
        else:
            print('Player reached goal.')
            print(board)
            break

        raw_input('Press enter to perform the action.')

        player, reward = gw.perform_action(player, action)
        print('\nReceived reward: {}'.format(reward))

