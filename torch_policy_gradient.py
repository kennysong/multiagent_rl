'''
    This is a policy gradient implementation (REINFORCE with v(s) baseline)
    on the two-agent Gridworld Cliff environment.
'''

import gridworld
import numpy as np
import random
import torch

from torch.autograd import Variable

# To run on GPU, change this boolean to True
cuda = False

# def run_episode(policy_net, gamma=1):
def run_episode(gamma=1):
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
        # action, probs = run_policy_network(policy_net, state)
        action, probs = random.choice(gridworld.action_space), 0

        # Take that action, then environment gives us the next state and reward
        next_s, reward = gridworld.perform_action(state, action)

        # Record [(state, action, probs), reward]
        episode.append([(state, action, probs), reward])
        state = next_s

        # This is taking ages
        if len(episode) > 100:
            break

    # We have the reward from each (state, action), now calculate the return
    T = len(episode)
    for i in range(T):
        ret = sum(gamma**(j-i) * episode[j][1] for j in range(i, T))
        episode[i].append(ret)

    return episode

def build_value_network():
    '''Builds an MLP value function approximator, which maps states to scalar
       values. It has one hidden layer with 32 units and tanh activations.
    '''
    layers = [2, 32, 1]
    value_net = torch.nn.Sequential(
                  torch.nn.Linear(layers[0], layers[1]),
                  torch.nn.Tanh(),
                  torch.nn.Linear(layers[1], layers[2])
                )
    if cuda: value_net.cuda()
    return value_net

def train_value_network(value_net, episode):
    '''Trains an MLP value function approximator based on the output of one
       episode. The value network will map states to scalar values.

       Parameters:
       episode is an list of episode data, see run_episode()

       Returns:
       The scalar loss of the newly trained value network.
    '''
    # Parse episode data into Numpy arrays of states and returns
    states = Variable(torch.Tensor([t[0][0] for t in episode]))
    returns = Variable(torch.Tensor([t[2] for t in episode]))

    if cuda:
        states = states.cuda()
        returns = returns.cuda()

    # Define loss function and optimizer
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.RMSprop(value_net.parameters(), lr=1e-3, eps=1e-5)

    # Train the value network on states, returns
    optimizer.zero_grad()
    pred_returns = value_net(states)
    loss = loss_fn(pred_returns, returns)
    loss.backward()
    optimizer.step()

    return loss.data[0]

def run_value_network(value_net, state):
    '''Wrapper function to feed one state into the given value network and
       return the value as a torch.Tensor.'''
    if cuda:
        result = value_net(Variable(torch.Tensor([state])).cuda())
    else:
        result = value_net(Variable(torch.Tensor([state])))
    return result.data

# TODO: policy network on GPU

def build_policy_network():
    '''Builds an LSTM policy network, which maps states to action vectors.

       More precisely, the input into the LSTM will be a 5-D vector consisting
       of [prev_output, state]. The output of the LSTM will be a 3-D vector that
       gives softmax probabilities of each action for the agents. This model
       only handles one time step, i.e. one agent, so it must be manually
       re-run for every agent.
    '''

    class PolicyNet(torch.nn.Module):
        def __init__(self, layers):
            super(PolicyNet, self).__init__()
            self.lstm = torch.nn.LSTMCell(layers[0], layers[1])
            self.linear = torch.nn.Linear(layers[1], layers[2])
            self.softmax = torch.nn.Softmax()

        def forward(self, x, h0, c0):
            h1, c1 = self.lstm(x, (h0, c0))
            o1 = self.softmax(self.linear(h1))
            return o1, h1, c1

    layers = [5, 32, 3]
    policy_net = PolicyNet(layers)
    return policy_net

def run_policy_network(policy_net, state):
    '''Wrapper function to feed a given state into the given policy network and
       return the action [a_v, a_h], as well as the softmax probability of each
       action [p_v, p_h].

       The initial input into the LSTM will be concat([0, 0, 0], state). This
       will output the softmax probabilities for the 3 vertical actions. We
       select one as a_v, a one-hot vector. The second input into the LSTM will
       be concat(a_v, state), which will output softmax probabilities for the 3
       horizontal actions. We select one as a_h.

       The output action [a_v, a_h] is transformed into a coordinate
       action vector, e.g. [-1, 1], instead of the one-hot vectors.
    '''

    actions = [-1, 0, 1]

    # TODO: What should h0, c0 be?
    # Predict action for the vertical agent and its probability
    initial_input = Variable(torch.Tensor(
                        np.append(np.zeros(3), state).reshape(1, 5)
                    ))
    h0, c0 = Variable(torch.zeros(1, 32)), Variable(torch.zeros(1, 32))
    dist_v, h1, c1 = policy_net(initial_input, h0, c0)
    index_v = np.random.choice(range(len(dist_v[0])), p=dist_v[0].data.numpy())
    p_v = dist_v.data[0][index_v]
    a_v = actions[index_v]

    # Convert a_v to a one-hot vector
    onehot_v = np.zeros(len(dist_v[0]))
    onehot_v[index_v] = 1

    # Predict action for the horizontal agent and its probability
    second_input = Variable(torch.Tensor(
                       np.append(onehot_v, state).reshape(1, 5)
                   ))
    dist_h, _, _ = policy_net(second_input, h1, c1)
    index_h = np.random.choice(range(len(dist_h[0])), p=dist_h[0].data.numpy())
    p_h = dist_h.data[0][index_h]
    a_h = actions[index_v]

    return np.array([a_v, a_h]), np.array([p_v, p_h])

