'''
    This is a policy gradient implementation (REINFORCE with v(s) baseline)
    on the two-agent Gridworld Cliff environment.
'''

import gridworld
import numpy as np
import random
import theano
import theano.tensor as T

from keras.models import Sequential
from keras.layers import Dense

def run_episode(policy):
    '''Runs one episode of Gridworld Cliff to completion with a policy, which
       is a function mapping states to actions.

       Returns:
       [((s_0, a_0), r_1), ..., ((s_{T-1}, a_{T-1}), r_T)]
         (s_t, a_t) is each state-action pair visited during the episode.
         r_{t+1} is the reward received from that state-action pair.
    '''
    # Initialize state as player position
    state = gridworld.start
    rewards = []

    # Run Gridworld until episode terminates at the goal
    while not np.array_equal(state, gridworld.goal):
        # Let our agent decide that to do at this state
        action = policy(state)

        # Take that action, then environment gives us the next state and reward
        next_s, reward = gridworld.perform_action(state, action)

        # Record ((state, action), reward)
        rewards.append(((state, action), reward))
        state = next_s

    # We have the reward from each (state, action), now calculate the return
    # returns, T = [], len(rewards)
    # for i in range(T):
    #     sa, ret = rewards[i][0], 0
    #     for j in range(i, T):
    #          ret += (gamma**(j-i)) * rewards[j][1]
    #     returns.append((sa, ret))

    # return returns
    return rewards

def random_policy(state):
    '''Returns a random action at any state.'''
    return random.choice(gridworld.action_space)

if __name__ == '__main__':
    returns = run_episode(random_policy)
    print(len(returns))
    print(returns[0])
    print(returns[1])
    print(returns[-1])
