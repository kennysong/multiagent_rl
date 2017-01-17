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

def run_episode(policy_net, gamma=1.0):
    '''Runs one episode of Gridworld Cliff to completion with a policy network,
       which is a LSTM that mapping states to actions, and returns the
       probabilities of those actions. gamma is the discount factor.

       Returns:
       [[(s_0, a_0, p_0), r_1, G_1], ..., [(s_{T-1}, a_{T-1}, p_{T-1}), r_T, G_T]]
         s_t, a_t is each state-action pair visited during the episode.
         p_t is the probability of taking a_t from s_t, given by the policy.
         r_{t+1} is the reward received from that state-action pair.
         G_{t+1} is the discounted return received from that state-action pair.
    '''
    # Initialize state as player position
    state = gridworld.start
    episode = []

    # Run Gridworld until episode terminates at the goal
    while not np.array_equal(state, gridworld.goal):
        # Let our agent decide that to do at this state
        action, probs = run_policy_network(policy_net, state)

        # Take that action, then environment gives us the next state and reward
        next_s, reward = gridworld.perform_action(state, action)

        # Record [(state, action, probs), reward]
        episode.append([(state, action, probs), reward])
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
       episode is an list of episode data, see run_episode()

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

       So, the LSTM has 5 input nodes and 3 output nodes.
    '''
    layers = [5, 32, 3]
    model = Sequential()

    # We use a stateful LSTM layer to be able to predict each agent's action
    # incrementally, by feeding 1 input with 1 time step of layers[0] features,
    # which will output one agent's action
    model.add(LSTM(layers[1], batch_input_shape=(1, 1, layers[0]), stateful=True))
    model.add(Dense(layers[2], activation='softmax'))

    # CHECK: We never use this loss function or optimizer, right? What should
    # it be set to?
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train_policy_network(model, episode, baseline=None, alpha=0.001):
    '''Update the policy network parameters with the REINFORCE algorithm.

       For each parameter W of the policy network, we make the following update:
         W += alpha * [grad_W(LSTM(a_t | s_t)) * (G_t - baseline(s_t))]
            = alpha * [grad_W(sum(log(p_t))) * (G_t - baseline(s_t))]
            = alpha * [sum(grad_W(log(p_t))) * (G_t - baseline(s_t))]
       for all time steps in the episode.

       Parameters:
       model is our LSTM policy network
       episode is an list of episode data, see run_episode()
       baseline is our MLP value network
    '''
    # CHECK: How do I fix this? 
    log_p_t = T.log(model.layers[1].output)
    W_i = model.weights[0]
    grads = T.grad(log_p_t, W_i)

    for data in episode:
        s_t = data[0][0]
        G_t = data[2]

        # CHECK: Somehow calculate gradients here
        # for W in model.weights:
            # CHECK: Something like this?
            # W.set_value(W.get_value() + alpha * gradient * (G_t - baseline))

def run_policy_network(model, state):
    '''Wrapper function to feed a given state into the given policy network and
       return the action [a_v, a_h], as well as the softmax probability of each
       action [p_v, p_h].

       The initial input into the LSTM will be concat([0, 0, 0], state). This
       will output the softmax probabilities for the 3 vertical actions. We
       select one as a_v, a one-hot vector. The second input into the LSTM will
       be concat(a_v, state), which will output softmax probabilities for the 3
       horizontal actions. We select one as a_h.

       For simplicity, the output action [a_v, a_h] is transformed into a valid
       action vector, e.g. [-1, 1], instead of the one-hot vectors.
    '''
    # Model is a stateful LSTM, so make sure the state is reset
    assert model.stateful
    model.reset_states()

    # Predict action for the vertical agent and its probability
    actions = [-1, 0, 1]
    initial_input = np.concatenate((np.zeros(3), state)).reshape(1, 1, 5)
    dist_v = model.predict(initial_input)[0]
    index_v = np.random.choice(range(len(dist_v)), p=dist_v)
    p_v = dist_v[index_v]
    a_v = actions[index_v]

    # Convert a_v to a one-hot vector
    onehot_v = np.zeros(len(dist_v))
    onehot_v[index_v] = 1

    # Predict action for the horizontal agent and its probability
    second_input = np.concatenate((onehot_v, state)).reshape(1, 1, 5)
    dist_h = model.predict(initial_input)[0]
    index_h = np.random.choice(range(len(dist_h)), p=dist_h)
    p_h = dist_h[index_h]
    a_h = actions[index_h]

    # Reset the states of the LSTM
    model.reset_states()

    return np.array([a_v, a_h]), np.array([p_v, p_h])

if __name__ == '__main__':
    policy_net = build_policy_network()
    episode = run_episode(policy_net)
    # value_net = train_value_network(episode)

    # print(run_policy_network(policy_net, np.array([3, 0])))
    train_policy_network(policy_net, episode)
