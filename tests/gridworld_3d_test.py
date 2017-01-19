'''
    Very basic testing for gridworld.py by selecting random actions and
    printing out the game state. You may want to decrease gridworld.grid_cols
    so the random agent can find the goal in a reasonable time.
'''

# Add parent directory to PYTHONPATH
# (http://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import gridworld_3d as gw
import numpy as np
import random

if __name__ == '__main__':
    player = gw.start
    reward = 0

    while True:
        print('\nReceived reward: {}'.format(reward))
        board = np.zeros((gw.grid_z, gw.grid_y, gw.grid_x))
        board[player[0], player[1], player[2]] = 1

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

        # Clears and overwrites previous lines
        sys.stdout.write('\033[K\033[F' * 45)
