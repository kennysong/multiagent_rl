'''
    Very basic testing for levers.py by selecting random actions and
    printing out the game state.
'''

# Add parent directory to PYTHONPATH
# (http://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import levers
import numpy as np

if __name__ == '__main__':
    state = levers.initial_state()
    print('State: {}'.format(state))

    action = np.random.randint(0, levers.m, size=levers.N)
    print('Random action: {}'.format(action))

    raw_input('Press enter to perform the action.')
    reward = levers.perform_action(state, action)
    print('Reward: {}'.format(reward))
