'''
    This is a policy gradient implementation (REINFORCE with v(s) baseline)
    on the two-agent Gridworld Cliff environment.

    Games should conform to the interface:
      game.num_agents - number of agents
      game.start_state() - returns start state of the game
      game.is_end(state) - given a state, return if the game/episode has ended
      game.perform_action(s, a) - given action indices at state s, returns next_s, reward
      game.set_options(options) - set options for the game
'''

import numpy as np
import random
import torch
import sys

from namedlist import namedlist
from torch.autograd import Variable

# TODO(Martin): Review GPU code, is running suspiciously slowly
# To run on GPU, change `cuda` to True
cuda = False
if cuda: print('Running policy gradient on GPU.')
FloatTensor = lambda x: torch.cuda.FloatTensor(x) if cuda else torch.FloatTensor(x)
ZeroTensor = lambda *s: torch.cuda.FloatTensor(*s).zero_() if cuda else torch.zeros(*s)

def run_episode(policy_net, gamma=1):
    '''Runs one episode of Gridworld Cliff to completion with a policy network,
       which is a LSTM that mapping states to actions, and returns the
       probabilities of those actions. gamma is the discount factor.

       Returns:
       [EpisodeStep(t=0), ..., EpisodeStep(t=T)]
    '''
    # Define a EpisodeStep container for each step in the episode:
    #   s, a is the state-action pair visited during that step
    #   grad_W is gradient term sum(grad_W(log(p))); see train_policy_net()
    #   r is the reward received from that state-action pair
    #   G is the discounted return received from that state-action pair
    EpisodeStep = namedlist('EpisodeStep', 's a grad_W r G', default=0)

    # Initialize state as player position
    state = game.start_state()
    episode = []

    # Run game until agent reaches the end
    while not game.is_end(state):
        # Let our agent decide that to do at this state
        a_indices, grad_W = run_policy_net(policy_net, state)

        # Take that action, then the game gives us the next state and reward
        next_s, r = game.perform_action(state, a_indices)

        # Record state, action, grad_W, reward
        episode.append(EpisodeStep(s=state, a=a_indices, grad_W=grad_W, r=r))
        state = next_s

        # This is taking ages
        if len(episode) > 100: break

    # We have the reward from each (state, action), now calculate the return
    T = len(episode)
    for i, step in enumerate(episode):
        step.G = sum(gamma**(j-i) * episode[j].r for j in range(i, T))

    return episode

def build_value_net(layers):
    '''Builds an MLP value function approximator, which maps states to scalar
       values. It has one hidden layer with 32 units and tanh activations.
    '''
    value_net = torch.nn.Sequential(
                  torch.nn.Linear(layers[0], layers[1]),
                  torch.nn.Tanh(),
                  torch.nn.Linear(layers[1], layers[2]))
    return value_net.cuda() if cuda else value_net

def train_value_net(value_net, episode):
    '''Trains an MLP value function approximator based on the output of one
       episode. The value network will map states to scalar values.

       Parameters:
       episode is a list of EpisodeStep's

       Returns:
       The scalar loss of the newly trained value network.
    '''
    # Parse episode data into Numpy arrays of states and returns
    states = Variable(FloatTensor([step.s for step in episode]))
    returns = Variable(FloatTensor([step.G for step in episode]))

    # Define loss function and optimizer
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.RMSprop(value_net.parameters(), lr=1e-3, eps=1e-5)

    # Train the value network on states, returns
    optimizer.zero_grad()
    loss = loss_fn(value_net(states), returns)
    loss.backward()
    optimizer.step()

    return loss.data[0]

def run_value_net(value_net, state):
    '''Wrapper function to feed one state into the given value network and
       return the value as a scalar.'''
    result = value_net(Variable(FloatTensor([state])))
    return result.data[0][0]

def build_policy_net(layers):
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
            self.layers = layers

        def forward(self, x, h0, c0):
            h1, c1 = self.lstm(x, (h0, c0))
            o1 = self.softmax(self.linear(h1))
            return o1, h1, c1

    policy_net = PolicyNet(layers)
    return policy_net.cuda() if cuda else policy_net

def run_policy_net(policy_net, state):
    '''Wrapper function to feed a given state into the given policy network and
       return an action vector, as well as parameter gradients.

       Essentially, run the policy_net LSTM for game.num_agents time-steps, to
       get the action for each agent, conditioned on previous agents' actions.
       As such, the input to the LSTM at time-step n is concat(a_{n-1}, state).
       We return a list of action indices, one index per agent.

       For each parameter W, the gradient term `sum(grad_W(log(p)))` is also
       computed and returned. This is used in the REINFORCE algorithm; see
       train_policy_net().
    '''
    # TODO(Martin): What should h_0, c_0 be?
    # Prepare initial inputs for policy_net
    h_size, a_size = policy_net.layers[1], policy_net.layers[2]
    a_indices = []
    a_n = np.zeros(a_size)
    h_n, c_n = Variable(ZeroTensor(1, h_size)), Variable(ZeroTensor(1, h_size))
    policy_net.zero_grad()

    # Use policy_net to predict output for each agent
    for n in range(game.num_agents):
        # TODO(Martin): Why is renormalizing flat_dist necessary on CUDA?
        # Predict action for the agent
        x_n = Variable(FloatTensor([np.append(a_n, state)]))
        dist, h_nn, c_nn = policy_net(x_n, h_n, c_n)
        flat_dist = np.array(dist[0].data.tolist())
        flat_dist /= sum(flat_dist)
        a_index = np.random.choice(range(a_size), p=flat_dist)

        # Calculate grad_W(log(p)), for all parameters W
        log_p = dist[0][a_index].log()
        log_p.backward()  # This will accumulate the gradient over all iterations

        # Record action for this iteration/agent
        a_indices.append(a_index)

        # Prepare inputs for next iteration/agent
        h_n, c_n = Variable(h_nn.data), Variable(c_n.data)
        a_n = np.zeros(a_size)
        a_n[a_index] = 1

    # Get the gradients; clone() is needed as the parameter Tensors are reused
    grad_W = [W.grad.data.clone() for W in policy_net.parameters()]

    return a_indices, grad_W

def train_policy_net(policy_net, episode, baseline=None, lr=3*1e-3):
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
       episode is an list of EpisodeStep's
       baseline is our MLP value network
    '''
    # Accumulate the update terms for each step in the episode into w_step
    W_step = [ZeroTensor(W.size()) for W in policy_net.parameters()]
    for step in episode:
        s_t, G_t, grad_W = step.s, step.G, step.grad_W
        for i in range(len(W_step)):
            if baseline:
                W_step[i] += grad_W[i] * (G_t - baseline(s_t))
            else:
                W_step[i] += grad_W[i] * G_t

    # Do a step of rprop
    for i, W in enumerate(policy_net.parameters()):
        W.data += lr * W_step[i] / (W_step[i].abs() + 1e-5)

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'gridworld':
        import gridworld as game
        policy_net_layers = [5, 32, 3]
        value_net_layers = [2, 32, 1]
    elif len(sys.argv) == 2 and sys.argv[1] == 'hunters':
        import hunters as game
        policy_net_layers = [9+8, 64, 9]
        value_net_layers = [8, 32, 1]
    else:
        sys.exit('Usage: python policy_gradient.py {gridworld, hunters}')

    policy_net = build_policy_net(policy_net_layers)
    value_net = build_value_net(value_net_layers)
    baseline = lambda state: run_value_net(value_net, state)

    cum_value_error, cum_return = 0.0, 0.0
    for num_episode in range(50000):
        episode = run_episode(policy_net)
        value_error = train_value_net(value_net, episode)
        cum_value_error = 0.9 * cum_value_error + 0.1 * value_error
        cum_return = 0.9 * cum_return + 0.1 * episode[0].G
        print("Num episode:{} Episode Len:{} Return:{} Baseline error:{}".format(num_episode, len(episode), cum_return, cum_value_error))
        train_policy_net(policy_net, episode, baseline=baseline)
