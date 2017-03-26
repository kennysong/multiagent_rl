'''
    This is a policy gradient implementation (REINFORCE with v(s) baseline)
    on the two-agent Gridworld Cliff environment.

    Games should conform to the interface:
      game.num_agents - number of agents
      game.start_state() - returns start state of the game
      game.is_end(state) - given a state, return if the game/episode has ended
      game.perform_action(s, a) - given action indices at state s, returns next_s, reward
      game.filter_actions(s, n) - filter actions available for an agent in a given state
      game.set_options(options) - set options for the game
'''

import argparse
import numpy as np
import os
import random
import sys
import torch

from namedlist import namedlist
from torch.autograd import Variable

# Define a EpisodeStep container for each step in an episode:
#   s, a is the state-action pair visited during that step
#   grad_W is gradient term sum(grad_W(log(p))); see train_policy_net()
#   r is the reward received from that state-action pair
#   G is the discounted return received from that state-action pair
EpisodeStep = namedlist('EpisodeStep', 's a grad_W r G', default=0)

def run_episode(policy_net, gamma=1):
    '''Runs one episode of Gridworld Cliff to completion with a policy network,
       which is a LSTM that mapping states to actions, and returns the
       probabilities of those actions. gamma is the discount factor.

       Returns:
       [EpisodeStep(t=0), ..., EpisodeStep(t=T)]
    '''
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

        # Terminate episode early
        if len(episode) > max_episode_len:
            episode[-1].r += max_len_penalty
            break

    # We have the reward from each (state, action), now calculate the return
    for i, step in enumerate(reversed(episode)):
        if i == 0: step.G = step.r
        else: step.G = step.r + gamma*episode[len(episode)-i].G

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
       episode, i.e. first-visit Monte-Carlo policy evaluation. The value
       network will map states to scalar values.

       Warning: currently only works for integer-vector states!

       Parameters:
       episode is a list of EpisodeStep's

       Returns:
       The scalar loss of the newly trained value network.
    '''
    # Calculate return from the first visit to each state
    visited_states = set()
    states, returns = [], []
    for i in range(len(episode)):
        s, G = episode[i].s, episode[i].G
        str_s = s.astype(int).tostring()  # Fastest hashable state representation
        if str_s not in visited_states:
            visited_states.add(str_s)
            states.append(s)
            returns.append(G)
    states = Variable(FloatTensor(states))
    returns = Variable(FloatTensor(returns))

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

       More precisely, the input into the LSTM will be a vector consisting of
       [prev_output, state]. The output of the LSTM will be a vector that
       gives softmax probabilities of each action for the agents. This model
       only handles one time step, i.e. one agent, so it must be manually
       re-run for every agent.
    '''
    class PolicyNet(torch.nn.Module):
        def __init__(self, layers):
            super(PolicyNet, self).__init__()
            self.lstm = torch.nn.LSTMCell(layers[0], layers[1])
            self.linear = torch.nn.Linear(layers[1], layers[2])
            self.layers = layers

        def forward(self, x, h0, c0):
            h1, c1 = self.lstm(x, (h0, c0))
            o1 = self.linear(h1)
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

       For each parameter W, the gradient term `grad_W(sum(log(p)))` is also
       computed and returned. This is used in the REINFORCE algorithm; see
       train_policy_net().
    '''
    # TODO(Martin): What should h_0, c_0 be?
    # Prepare initial inputs for policy_net
    global h_n, c_n, sum_log_p, h_size, a_size
    a_indices = []
    a_n = np.zeros(a_size)
    h_n.data.zero_(); c_n.data.zero_()
    sum_log_p.detach_(); sum_log_p.data.zero_()
    policy_net.zero_grad()
    softmax = torch.nn.Softmax()

    # Use policy_net to predict output for each agent
    for n in range(game.num_agents):
        # Do a forward step through policy_net, filter actions, and softmax it
        x_n = Variable(FloatTensor([np.append(a_n, state)]))
        o_nn, h_nn, c_nn = policy_net(x_n, h_n, c_n)
        action_mask = ByteTensor(game.filter_actions(state, n))
        filt_o_nn = o_nn[action_mask].resize(1, action_mask.sum())
        dist = softmax(filt_o_nn)

        # Randomly sample an available action from dist
        filt_a = np.arange(a_size)[action_mask.cpu().numpy().astype(bool)]
        a_index = np.random.choice(filt_a, p=dist[0].data.cpu().numpy())

        # Calculate sum(log(p))
        filt_a_index = 0 if a_index == 0 else action_mask[:a_index].sum()
        log_p = dist[0][filt_a_index].log()
        sum_log_p += log_p

        # Record action for this iteration/agent
        a_indices.append(a_index)

        # Prepare inputs for next iteration/agent
        h_n, c_n = Variable(h_nn.data), Variable(c_nn.data)
        a_n = np.zeros(a_size)
        a_n[a_index] = 1

    # Get the gradients; clone() is needed as the parameter Tensors are reused
    sum_log_p.backward()
    grad_W = [W.grad.data.clone() for W in policy_net.parameters()]

    return a_indices, grad_W

def train_policy_net(policy_net, episode, baseline=None, td=None, lr=3*1e-3,
                     opt='rmsprop'):
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
       baseline is function mapping states to values
       td is the k for a TD(k) gradient term, td=None for a Monte-Carlo term
       opt is the optimizer to use, either 'rmsprop' or 'rprop'
    '''
    # Calculate baselines, if being used
    if baseline:
        baselines = [baseline(step.s) for step in episode]

    # Accumulate the update terms for each step in the episode into W_step
    for W in W_step: W.zero_()
    for t, step in enumerate(episode):
        s_t, G_t, r_t, grad_W = step.s, step.G, step.r, step.grad_W
        for i in range(len(W_step)):
            # Monte-Carlo baselined update
            if baseline and td is None:
                W_step[i] += grad_W[i] * (G_t - baselines[t])
            # TD(0) baselined update
            elif baseline and td == 0:
                if t == len(episode)-1:
                    W_step[i] += grad_W[i] * (r_t - baselines[t])
                else:
                    W_step[i] += grad_W[i] * (r_t + baselines[t+1] - baselines[t])
            # TD(1) baselined update
            elif baseline and td == 1:
                if t == len(episode)-1:
                    W_step[i] += grad_W[i] * (r_t - baselines[t])
                elif t == len(episode)-2:
                    # W_step[i] += grad_W[i] * (r_t + baselines[t+1] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r - baselines[t])
                else:
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + baselines[t+2] - baselines[t])
            # TD(2) baselined update
            elif baseline and td == 2:
                if t == len(episode)-1:
                    W_step[i] += grad_W[i] * (r_t - baselines[t])
                elif t == len(episode)-2:
                    # W_step[i] += grad_W[i] * (r_t + baselines[t+1] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r - baselines[t])
                elif t == len(episode)-3:
                    # W_step[i] += grad_W[i] * (r_t + episode[t+1].r + baselines[t+2] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r - baselines[t])
                else:
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r + baselines[t+3] - baselines[t])
            # TD(3) baselined update
            elif baseline and td == 3:
                if t == len(episode)-1:
                    W_step[i] += grad_W[i] * (r_t - baselines[t])
                elif t == len(episode)-2:
                    # W_step[i] += grad_W[i] * (r_t + baselines[t+1] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r - baselines[t])
                elif t == len(episode)-3:
                    # W_step[i] += grad_W[i] * (r_t + episode[t+1].r + baselines[t+2] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r - baselines[t])
                elif t == len(episode)-4:
                    # W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r + baselines[t+3] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r + episode[t+3].r - baselines[t])
                else:
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r + episode[t+3].r + baselines[t+4] - baselines[t])
            # TD(4) baselined update
            elif baseline and td == 4:
                if t == len(episode)-1:
                    W_step[i] += grad_W[i] * (r_t - baselines[t])
                elif t == len(episode)-2:
                    # W_step[i] += grad_W[i] * (r_t + baselines[t+1] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r - baselines[t])
                elif t == len(episode)-3:
                    # W_step[i] += grad_W[i] * (r_t + episode[t+1].r + baselines[t+2] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r - baselines[t])
                elif t == len(episode)-4:
                    # W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r + baselines[t+3] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r + episode[t+3].r - baselines[t])
                elif t == len(episode)-5:
                    # W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r + episode[t+3].r + baselines[t+4] - baselines[t])
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r + episode[t+3].r + episode[t+4].r - baselines[t])
                else:
                    W_step[i] += grad_W[i] * (r_t + episode[t+1].r + episode[t+2].r + episode[t+3].r + episode[t+4].r + baselines[t+5] - baselines[t])
            # Monte-Carlo update without baseline
            else:
                W_step[i] += grad_W[i] * G_t

    if opt == 'rprop':  # Do a step of rprop
        for i, W in enumerate(policy_net.parameters()):
            W.data += lr * W_step[i] / (W_step[i].abs() + 1e-5)
    elif opt == 'rmsprop':  # Do a step of RMSprop
        eps = 1e-5  # For numerical stability
        alpha = 0.9  # Weighted average factor
        for i in range(len(W_step)):
            mean_square[i] = alpha*mean_square[i] + (1-alpha)*W_step[i].pow(2)
            W_step[i] = lr * W_step[i] / (mean_square[i] + eps).sqrt()
        for i, W in enumerate(policy_net.parameters()):
            W.data += W_step[i]

def set_options(options):
    '''Sets policy gradient options.'''
    global cuda, max_episode_len, max_len_penalty, FloatTensor, ZeroTensor, ByteTensor
    cuda = options.cuda
    max_episode_len = options.max_episode_len
    max_len_penalty = options.max_len_penalty
    FloatTensor = lambda x: torch.cuda.FloatTensor(x) if cuda else torch.FloatTensor(x)
    ZeroTensor = lambda *s: torch.cuda.FloatTensor(*s).zero_() if cuda else torch.zeros(*s)
    ByteTensor = lambda x: torch.cuda.ByteTensor(x) if cuda else torch.ByteTensor(x)
    if cuda: print('Running policy gradient on GPU.')

    # Transparently set number of threads based on environment variables
    num_threads = int(os.getenv('OMP_NUM_THREADS', 1))
    torch.set_num_threads(num_threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs multi-agent policy gradient.')
    parser.add_argument('--game', choices=['gridworld', 'gridworld_3d', 'hunters'], required=True, help='A game to run')
    parser.add_argument('--cuda', action='store_true', default=False, help='Include to run on CUDA')
    parser.add_argument('--max_episode_len', default=float('inf'), type=float, help='Terminate episode early at this number of steps')
    parser.add_argument('--max_len_penalty', default=0, type=float, help='If episode is terminated early, add this to the last reward')
    parser.add_argument('--num_episodes', default=10000, type=int, help='Number of episodes to run in a round of training')
    parser.add_argument('--num_rounds', default=1, type=int, help='How many rounds of training to run')
    parser.add_argument('--policy_net_opt', default='rmsprop', choices=['rmsprop', 'rprop'], help='Optimizer for training the policy net')
    parser.add_argument('--policy_net_update', choices=[0, 1, 2, 3, 4], type=int, help='k for a TD(k) gradient term; exclude for a Monte-Carlo update')
    args = parser.parse_args()
    set_options(args)

    if args.game == 'gridworld':
        import gridworld as game
        policy_net_layers = [5, 32, 3]
        value_net_layers = [2, 32, 1]
        game.set_options({'grid_y': 7, 'grid_x': 7})
    elif args.game == 'gridworld_3d':
        import gridworld_3d as game
        policy_net_layers = [6, 32, 3]
        value_net_layers = [3, 32, 1]
        game.set_options({'grid_z': 4, 'grid_y': 4, 'grid_x': 4})
    elif args.game == 'hunters':
        import hunters as game
        policy_net_layers = [17, 128, 9]
        value_net_layers = [8, 64, 1]
        game.set_options({'rabbit_action': None, 'remove_hunters': True,
                          'capture_reward': 10})

    for i in range(args.num_rounds):
        policy_net = build_policy_net(policy_net_layers)
        value_net = build_value_net(value_net_layers)
        baseline = lambda state: run_value_net(value_net, state)

        # Init main Tensors first, so we don't have to allocate memory at runtime
        # TODO: Check again after https://github.com/pytorch/pytorch/issues/339
        #   Used in run_policy_net():
        h_size, a_size = policy_net_layers[1], policy_net_layers[2]
        h_n, c_n = Variable(ZeroTensor(1, h_size)), Variable(ZeroTensor(1, h_size))
        x_n = Variable(ZeroTensor(1, policy_net_layers[0]))
        sum_log_p = Variable(ZeroTensor(1))
        #   Used in train_policy_net():
        W_step = [ZeroTensor(W.size()) for W in policy_net.parameters()]
        mean_square = [ZeroTensor(W.size()) for W in policy_net.parameters()]
        for W in mean_square: W += 1

        avg_value_error, avg_return = 0.0, 0.0
        for num_episode in range(args.num_episodes):
            episode = run_episode(policy_net)
            value_error = train_value_net(value_net, episode)
            avg_value_error = 0.9 * avg_value_error + 0.1 * value_error
            avg_return = 0.9 * avg_return + 0.1 * episode[0].G
            print("{{'i': {}, 'num_episode': {}, 'episode_len': {}, 'episode_return': {}, 'avg_return': {}, 'avg_value_error': {}}},".format(i, num_episode, len(episode), episode[0].G, avg_return, avg_value_error))
            train_policy_net(policy_net, episode, baseline=baseline, td=args.policy_net_update, opt=args.policy_net_opt)
