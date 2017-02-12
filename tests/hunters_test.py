'''
    Very basic testing for hunters.py by selecting random actions and
    printing out the game state.
'''

# Add parent directory to PYTHONPATH
# (http://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import hunters
import numpy as np

if __name__ == '__main__':
    state = hunters.initial_state()
    reward = 0
    is_end = False

    while True:
        print('Total reward: {}'.format(reward))
        print('State: {}'.format(state))

        board = np.zeros((hunters.n, hunters.n))
        for i in range(0, 2*hunters.k, 2):
            if (state[i], state[i+1]) != (-1, -1):
                board[state[i], state[i+1]] = 1  # Hunters
        for j in range(2*hunters.k, 2*hunters.k+2*hunters.m, 2):
            if (state[j], state[j+1]) != (-1, -1):
                board[state[j], state[j+1]] = 3  # Rabbits
        print(board)

        if is_end:
            print('Hunters captured all rabbits.')
            break

        action = np.random.randint(-1, 2, size=2*hunters.k)
        print('Action: {}'.format(action))

        raw_input('Press enter to perform the action.')
        state, r, is_end = hunters.perform_action(state, action)
        reward += r

        # Clears and overwrites previous lines
        sys.stdout.write('\033[K\033[F' * 10)
