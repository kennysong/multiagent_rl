'''
    This is a policy gradient implementation (REINFORCE with v(s) baseline)
    on the two-agent hunters task.
'''

import hunters
import numpy as np
import random
import theano
import theano.tensor as T
import keras

from keras.models import Sequential
from keras.layers import Dense, LSTM, Input

def run_episode(policy_net, gamma=1):
    '''Runs one episode of hunters to completion with a policy network,
       which is a LSTM that mapping states to actions, and returns the
       probabilities of those actions. gamma is the discount factor.

       Returns:
       [[(s_0, a_0, p_0), r_1, G_1], ..., [(s_{T-1}, a_{T-1}, p_{T-1}), r_T, G_T]]
         s_t, a_t is each state-action pair visited during the episode.
         p_t is the probability of taking a_t from s_t, given by the policy.
         r_{t+1} is the reward received from that state-action pair.
         G_{t+1} is the discounted return received from that state-action pair.
    '''
    # Initialize hunters environment state
    state = hunters.initial_state()
    is_end = False
    episode = []

    # Run hunters task until all rabbits are captured
    while not is_end:
        # Let our agent decide that to do at this state
        action, probs = run_policy_network(policy_net, state)

        # Take that action, then environment gives us the next state and reward
        next_s, reward, is_end = hunters.perform_action(state, action, remove_hunter=True, capture_reward=True)

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
       values. It has one hidden layer with 32 units and relu activations.
    '''
    layers = [8, 32, 1]
    model = Sequential()
    model.add(Dense(layers[1], input_dim=layers[0], activation='tanh')) # Relus throw nans.
    model.add(Dense(layers[2]))

    opt = keras.optimizers.RMSprop(lr=1e-3, epsilon=1e-5)
    model.compile(optimizer=opt, loss='mae') # MSE was throwing nans
    return model

def train_value_network(model, episode):
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
    error = model.train_on_batch(states, returns)

    return error

def run_value_network(model, state):
    '''Wrapper function to feed a given state into the given value network and
       return the value.'''
    result = model.predict(np.array([state]))
    return result[0][0]

def build_policy_network():
    '''Builds an LSTM policy network, which maps states to action vectors.

       More precisely, the input into the LSTM will be a 10-D vector consisting
       of state + prev_action. The output of the LSTM will be a 9-D vector that
       gives softmax probabilities of each action for the agents.

       So, the LSTM has 10 input nodes and 9 output nodes.
    '''
    layers = [10, 64, 9]
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

# TODO: make this not hardcoded to this architecture, input output dims and such
def compile_gradient_functions(model):
    '''Compiles a Theano function that calculates grad_W(sum(log(p_t))) for all
       parameters W of the LSTM, given specific inputs into the LSTM (input_1,
       input_2) and the selected actions (index_v, index_h).

       Parameters:
       model is our LSTM policy network
    '''
    index_h1 = T.iscalar()
    index_h2 = T.iscalar()

    input_1 = Input(shape=(1, 10,)) #TODO: Check that this is okay. I think it adds an extra batch_size dim to the first one so it's okay.
    input_2 = Input(shape=(1, 10,))

    dist_h1 = model(input_1)
    log_p_t_h1 = T.log(dist_h1[0, index_h1]) # First dimension is over samples in batch so = 1

    # TODO: I'm not massively sure the stateful is going through this but it should
    # Where's the symbolic 'model.reset_states'?
    dist_h2 = model(input_2)
    log_p_t_h2 = T.log(dist_h2[0, index_h2])

    log_p_t_sum = log_p_t_h1 + log_p_t_h2

    grads_log_p_t = [T.grad(log_p_t_sum, w) for w in model.weights]

    get_gradients = theano.function([input_1, input_2, index_h1, index_h2], grads_log_p_t, allow_input_downcast=True)

    return get_gradients

def train_policy_network(model, episode, get_gradients, baseline=None, lr=3*1e-3):
    '''Update the policy network parameters with the REINFORCE algorithm.

       For each parameter W of the policy network, we make the following update:
         W += alpha * [grad_W(LSTM(a_t | s_t)) * (G_t - baseline(s_t))]
            = alpha * [grad_W(sum(log(p_t))) * (G_t - baseline(s_t))]
       for all time steps in the episode.

       Parameters:
       model is our LSTM policy network
       episode is an list of episode data, see run_episode()
       baseline is our MLP value network
    '''

    w_step = [np.zeros(w.get_value().shape, dtype='float32') for w in model.weights]

    if baseline is not None:
        state_batch = np.asarray([data[0][0] for data in episode])
        baseline_predicts = baseline.predict(state_batch)
    for t, data in enumerate(episode):
        s_t = data[0][0]
        a_t = data[0][1]
        G_t = data[2]

        index_h1 = hunters.action_coordinates_to_index(a_t[:2])
        index_h2 = hunters.action_coordinates_to_index(a_t[2:])

        input_h1 = np.concatenate((np.zeros(2), s_t)).reshape(1, 1, 10)
        input_h2 = np.concatenate((a_t[:2], s_t)).reshape(1, 1, 10)

        # Is this the place to reset states?
        model.reset_states()
        gradients = get_gradients(input_h1, input_h2, index_h1, index_h2)
        model.reset_states() # Again to be safe

        for i in range(len(w_step)):
            if baseline == None:
                w_step[i] += gradients[i] * G_t
            else:
                w_step[i] += gradients[i] * (G_t - baseline_predicts[t])

    # TODO: do rmsprop instead of rprop
    for i, w in enumerate(model.weights):
        w.set_value(w.get_value() + lr * w_step[i] / (np.abs(w_step[i]) + 1e-5))

def run_policy_network(model, state):
    '''Wrapper function to feed a given state into the given policy network and
       return the (coordinate) action vector, as well as the softmax
       probability of each agent's action [p_h1, p_h2].

       The initial input into the LSTM will be concat([0, 0], state). This
       will output the softmax probabilities for the 9 possible actions of the
       first hunter. We sample one as a_h1, a coordinate action vector. The
       second input into the LSTM will be concat(a_h1, state), which will
       output softmax probabilities for the 9 possible actions of the second
       hunter. We sample one as a_h2.

       For simplicity, the output action [a_v, a_h] is transformed into a valid
       action vector, e.g. [-1, 1], instead of the one-hot vectors.
    '''

    # Model is a stateful LSTM, so make sure the state is reset
    assert model.stateful
    model.reset_states()

    # Predict action for the first hunter and its probability
    initial_input = np.concatenate((np.zeros(2), state)).reshape(1, 1, 10)
    dist_h1 = model.predict(initial_input)[0]
    index_h1 = np.random.choice(range(len(dist_h1)), p=dist_h1)
    p_h1 = dist_h1[index_h1]
    a_h1 = hunters.action_index_to_coordinates(index_h1)

    # Predict action for the horizontal agent and its probability
    second_input = np.concatenate((a_h1, state)).reshape(1, 1, 10)
    dist_h2 = model.predict(second_input)[0]
    index_h2 = np.random.choice(range(len(dist_h2)), p=dist_h2)
    p_h2 = dist_h2[index_h2]
    a_h2 = hunters.action_index_to_coordinates(index_h2)

    # Reset the states of the LSTM
    model.reset_states()

    return np.concatenate((a_h1, a_h2)), np.array((p_h1, p_h2))

if __name__ == '__main__':
    assert hunters.k == 2 and hunters.m == 2  # Network architecture is hardcoded

    policy_net = build_policy_network()
    value_net = build_value_network()

    get_gradients = compile_gradient_functions(policy_net)
    cum_value_error = 0.0
    cum_return = 0.0
    for num_episode in range(50000):
        episode = run_episode(policy_net, gamma=1)
        value_error = train_value_network(value_net, episode)
        cum_value_error = 0.9 * cum_value_error + 0.1 * value_error
        cum_return = 0.9 * cum_return + 0.1 * episode[0][2]
        print("Num episode:{0} Len episode:{1} Return:{2} Baseline error:{3}".format(num_episode, len(episode), cum_return, cum_value_error)) # Print episode return
        train_policy_network(policy_net, episode, get_gradients, baseline=value_net)
