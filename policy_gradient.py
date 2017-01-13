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
from keras.layers import LSTM

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

def build_value_network():
    '''Builds an MLP value function approximator, which maps states to scalar
       values. It has one hidden layer with 10 units and relu activations.
    '''
    layers = [2, 10, 1]
    model = Sequential()
    model.add(Dense(layers[1], input_dim=layers[0], activation='relu'))
    model.add(Dense(layers[2]))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def train_value_network(episode):
    '''Trains an MLP value function approximator based on the output of one
       episode. The value network will map states to scalar values.

       Parameters:
       episode is [[(s_0, a_0), r_1, G_1], ..., [(s_{T-1}, a_{T-1}), r_T, G_T]]
         (s_t, a_t) is each state-action pair visited during the episode.
         r_{t+1} is the reward received from that state-action pair.
         G_{t+1} is the discounted return received from that state-action pair.

       Returns:
       The trained value network as a Keras Model.
    '''
    # Parse episode data into Numpy arrays of states and returns
    states = np.array([t[0][0] for t in episode])
    returns = np.array([t[2] for t in episode])

    # Train the MLP model on states, returns
    model = build_value_network()
    model.fit(states, returns, nb_epoch=1, verbose=0)

    return model

def run_value_network(model, state):
    '''Wrapper function to feed a given state into the given value network and
       return the value.'''
    result = model.predict(np.array([state]))
    return result[0][0]

def build_policy_network():
    '''Builds an LSTM policy network, which maps states to action vectors.

       More precisely, the input into the LSTM will be a 5-D vector consisting
       of prev_output + state. The output of the LSTM will be a 3-D vector that
       gives softmax probabilities of each action for the agents.

       So, the LSTM has 3 input nodes and 3 output nodes.
    '''
    # CHECK: What should the middle number be and what does it mean here?
    layers = [5, 1, 3]

    model = Sequential()
    model.add(LSTM(layers[1], input_dim=layers[0]))
    model.add(Dense(layers[2], activation='softmax'))

    # CHECK: We never use this loss function or optimizer, right? What should
    # it be set to?
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def run_policy_network(model, state):
    '''Wrapper function to feed a given state into the given policy network and
       return the action [a_v, a_h], as well as the softmax probability of each
       action [p_v, p_h].

       The initial input into the LSTM will be [0, 0, 0] + state. This will
       output the softmax probabilities for the 3 vertical actions. We select
       one as a_v, a one-hot vector. The second input into the LSTM will be
       a_v + state, which will output softmax probabilities for the 3
       horizontal actions. We select one as a_h.

       For simplicity, the output action [a_v, a_h] is transformed into a valid
       action vector, e.g. [-1, 1], instead of the one-hot vectors.
    '''
    # CHECK: this doesn't work, why?
    initial_input = np.concatenate((np.zeros(3), state)).reshape(1, 1, 5)
    dist_v = model.predict(initial_input)

def random_policy(state):
    '''Returns a random action at any state.'''
    return random.choice(gridworld.action_space)

if __name__ == '__main__':
    # episode = run_episode(random_policy)
    # value_net = train_value_network(episode)

    policy_net = build_policy_network()
    run_policy_network(policy_net, np.array([3, 0]))
