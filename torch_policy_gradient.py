'''
    This is a policy gradient implementation (REINFORCE with v(s) baseline)
    on the two-agent Gridworld Cliff environment.
'''

import gridworld
import numpy as np
import random
import torch

from torch.autograd import Variable

# TODO: Make policy_net work on GPU
# To run on GPU, change this boolean to True
cuda = False

def run_episode(policy_net, gamma=1):
    '''Runs one episode of Gridworld Cliff to completion with a policy network,
       which is a LSTM that mapping states to actions, and returns the
       probabilities of those actions. gamma is the discount factor.

       Returns:
       [
        [(s_0, a_0, grad_W_0), r_1, G_1], 
        ...
        [(s_{T-1}, a_{T-1}, grad_W_{T-1}), r_T, G_T]
       ]
         s_t, a_t is each state-action pair visited during the episode.
         grad_W_t is gradient term sum(grad_W(log(p))); see train_policy_network()
         r_{t+1} is the reward received from that state-action pair.
         G_{t+1} is the discounted return received from that state-action pair.
    '''
    # Initialize state as player position
    state = gridworld.start
    episode = []

    # Run Gridworld until episode terminates at the goal
    while not np.array_equal(state, gridworld.goal):
        # Let our agent decide that to do at this state
        action, grads = run_policy_network(policy_net, state)

        # Take that action, then environment gives us the next state and reward
        next_s, reward = gridworld.perform_action(state, action)

        # Record [(state, action, probs), reward]
        episode.append([(state, action, grads), reward])
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
                  torch.nn.Linear(layers[1], layers[2]))
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
       return the value as a scalar.'''
    if cuda:
        result = value_net(Variable(torch.Tensor([state])).cuda())
    else:
        result = value_net(Variable(torch.Tensor([state])))
    return result.data[0][0]

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
    if cuda: policy_net.cuda()
    return policy_net

def run_policy_network(policy_net, state):
    '''Wrapper function to feed a given state into the given policy network and
       return an action vector, as well as parameter gradients.

       Essentially, we run the policy_net LSTM for game.num_agents time-steps,
       in order to get the action for each agent, conditioned on the actions of
       the previous agents. As such, the input of the LSTM at time-step n is 
       concat(a_{n-1}, state).

       For each parameter W, the gradient term `sum(grad_W(log(p)))` is also 
       computed and returned. This is used in the REINFORCE algorithm; see
       train_policy_network().
    '''
    # TODO: What should h_0, c_0 be?
    # Prepare initial inputs for policy_net
    actions = [-1, 0, 1]
    a_n = np.zeros(3)
    h_n, c_n = Variable(torch.zeros(1, 32)), Variable(torch.zeros(1, 32))
    action = []
    grad_W = [torch.zeros(W.size()) for W in policy_net.parameters()]

    # Use policy_net to predict output for each agent
    for n in range(gridworld.num_agents):
        # Predict action for the agent
        x_n = Variable(torch.Tensor(np.append(a_n, state).reshape(1, 5)))
        dist, h_nn, c_nn = policy_net(x_n, h_n, c_n)
        a_index = np.random.choice(range(len(dist[0])), p=dist[0].data.numpy())

        # Calculate grad_W(log(p)), for all parameters W
        log_p = dist[0][a_index].log()
        policy_net.zero_grad()
        log_p.backward()
        grad_log_p = [W.grad.data for W in policy_net.parameters()]

        # Track output of this iteration/agent
        action.append(actions[a_index])
        for i in range(len(grad_W)): grad_W[i] += grad_log_p[i]

        # Prepare inputs for next iteration/agent
        h_n = Variable(h_nn.data)
        c_n = Variable(c_nn.data)
        a_n = np.zeros(3)
        a_n[a_index] = 1

    return np.array(action), grad_W

def train_policy_network(policy_net, episode, baseline=None, lr=3*1e-3):
    '''Update the policy network parameters with the REINFORCE algorithm.

       For each parameter W of the policy network, for each time-step t in the
       episode, make the update:
         W += alpha * [grad_W(LSTM(a_t | s_t)) * (G_t - baseline(s_t))]
            = alpha * [grad_W(sum(log(p))) * (G_t - baseline(s_t))]
            = alpha * [sum(grad_W(log(p))) * (G_t - baseline(s_t))]
       for all time steps in the episode.

       (Note: The sum is over the number of agents, each with an associated p)

       Parameters:
       model is our LSTM policy network
       episode is an list of episode data, see run_episode()
       baseline is our MLP value network
    '''
    # Accumulate the update terms for each step in the episode into w_step
    W_step = [torch.zeros(W.size()) for W in policy_net.parameters()]
    for t, data in enumerate(episode):
        s_t, G_t, grad_W = data[0][0], data[2], data[0][2]
        for i in range(len(W_step)):
            if baseline:
                W_step[i] += grad_W[i] * (G_t - baseline(s_t))
            else:
                W_step[i] += grad_W[i] * G_t

    # Do a step of rprop
    for i, W in enumerate(policy_net.parameters()):
        W.data += lr * W_step[i] / (W_step[i].abs() + 1e-5)

policy_net = build_policy_network()
value_net = build_value_network()
baseline = lambda state: run_value_network(value_net, state)

cum_value_error = 0.0
cum_return = 0.0
for num_episode in range(50000):
    episode = run_episode(policy_net, gamma=1)
    value_error = train_value_network(value_net, episode)
    cum_value_error = 0.9 * cum_value_error + 0.1 * value_error
    cum_return = 0.9 * cum_return + 0.1 * episode[0][2]
    print("Num episode:{} Episode Len:{} Return:{} Baseline error:{}".format(num_episode, len(episode), cum_return, cum_value_error)) # Print episode return
    train_policy_network(policy_net, episode, baseline=baseline)
