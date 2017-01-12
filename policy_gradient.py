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

def run_episode(policy, gamma=1.0):
    '''Runs one episode of Gridworld Cliff to completion with a policy, which
       is a function mapping states to actions. gamma is the discount factor.

       Returns:
       [[(s_0, a_0), r_1, G_1], ..., [(s_{T-1}, a_{T-1}), r_T, G_T]]
         (s_t, a_t) is each state-action pair visited during the episode.
         r_{t+1} is the reward received from that state-action pair.
         G_{t+1} is the discounted return received from that state-action pair.
    '''
    # Initialize state as player position
    state = gridworld.start
    episode = []

    # Run Gridworld until episode terminates at the goal
    while not np.array_equal(state, gridworld.goal):
        # Let our agent decide that to do at this state
        action = policy(state)

        # Take that action, then environment gives us the next state and reward
        next_s, reward = gridworld.perform_action(state, action)

        # Record [(state, action), reward]
        episode.append([(state, action), reward])
        state = next_s

    # We have the reward from each (state, action), now calculate the return
    T = len(episode)
    for i in range(T):
        ret = sum(gamma**(j-i) * episode[j][1] for j in range(i, T))
        episode[i].append(ret)

    return episode

def train_value_network(episode):
    '''Trains an MLP value function approximator based on the output of one
       episode. The value network will map states to scalar values.

       The MLP has one hidden layer with 128 units and relu activations.

       Parameters:
       episode is [[(s_0, a_0), r_1, G_1], ..., [(s_{T-1}, a_{T-1}), r_T, G_T]]
         (s_t, a_t) is each state-action pair visited during the episode.
         r_{t+1} is the reward received from that state-action pair.
         G_{t+1} is the discounted return received from that state-action pair.

       Returns:
       The value network as a Keras Model.
    '''
    # Parse episode data into Numpy arrays of states and returns
    states = np.array([t[0][0] for t in episode])
    returns = np.array([t[2] for t in episode])

    # Define MLP model
    layers = [2, 128, 1]
    model = Sequential()
    model.add(Dense(layers[1], input_dim=layers[0], activation='relu'))
    model.add(Dense(layers[2]))
    model.compile(optimizer='rmsprop', loss='mse')

    # Train the MLP model on states, returns
    model.fit(states, returns, nb_epoch=100, verbose=0)

    return model

def random_policy(state):
    '''Returns a random action at any state.'''
    return random.choice(gridworld.action_space)

if __name__ == '__main__':
    episode = run_episode(random_policy)
    model = train_value_network(episode)
