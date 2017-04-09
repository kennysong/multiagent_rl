'''
    This is a single-agent policy gradient implementation (REINFORCE with
    baselines). It is a baseline to compare multi-agent policy gradient to.
    Works for any game that conforms to the interface:

      game.start_state() - returns start state of the game
      game.is_end(state) - given a state, return if the game/episode has ended
      game.perform_joint_action(s, a) - given a joint action at state s, returns next_s, reward
      game.filter_joint_actions(s) - filter actions available in a given state
      game.set_options(options) - set options for the game
'''

import argparse
import numpy as np
import os
import random
import sys
import torch

from torch.autograd import Variable

import policy_gradient
from policy_gradient import EpisodeStep, build_value_net, run_value_net, \
                            train_value_net, train_policy_net

def run_episode(policy_net, gamma=1.0):
    '''Runs one episode of Gridworld Cliff to completion with a policy network,
       which is a MLP that maps states to joint action probabilities.

       Parameters:
       policy_net is our MLP policy network
       gamma is the discount factor for calculating returns

       Returns:
       [EpisodeStep(t=0), ..., EpisodeStep(t=T)]
    '''
    # Initialize state as player position
    state = game.start_state()
    episode = []

    # Run game until agent reaches the end
    while not game.is_end(state):
        # Let our agent decide that to do at this state
        a, grad_W = run_policy_net(policy_net, state)

        # Take that action, then the game gives us the next state and reward
        next_s, r = game.perform_joint_action(state, a)

        # Record state, action, grad_W, reward
        episode.append(EpisodeStep(s=state, a=a, grad_W=grad_W, r=r))
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

def build_policy_net(layers):
    '''Builds an MLP policy network, which maps states to action vectors.

       More precisely, the input into the MLP will be the state vector. The
       output of the MLP will be a vector that gives unnormalized probabilities
       of each joint action (one-hot vector of a combination of agents'
       actions). Softmax is applied afterwards, see run_policy_net().
    '''
    return build_value_net(layers)

def run_policy_net(policy_net, state):
    '''Wrapper function to feed a given state into the given policy network and
       return an action vector, as well as parameter gradients.

       Essentially, run the policy_net MLP to get a probability vector for all
       joint actions. Sample one, and denote its probability by p.

       For each parameter W, the gradient term `grad_W(log(p))` is also
       computed and returned. This is used in the REINFORCE algorithm; see
       train_policy_net().
    '''
    # Prepare for forward and backward pass
    a = [0] * a_size
    policy_net.zero_grad()
    softmax = torch.nn.Softmax()

    # Do a forward step through policy_net, filter actions, and softmax it
    x = Variable(FloatTensor([state]))
    o = policy_net(x)
    action_mask = ByteTensor(game.filter_joint_actions(state))
    filt_o = o[action_mask].resize(1, action_mask.sum())
    dist = softmax(filt_o)

    # Sample an available action from dist
    filt_a = np.arange(a_size)[action_mask.cpu().numpy().astype(bool)]
    a_index = np.random.choice(filt_a, p=dist[0].data.cpu().numpy())
    a[a_index] = 1

    # Calculate log(p)
    filt_a_index = 0 if a_index == 0 else action_mask[:a_index].sum()
    log_p = dist[0][filt_a_index].log()

    # Get the gradients; clone() is needed as the parameter Tensors are reused
    log_p.backward()
    grad_W = [W.grad.data.clone() for W in policy_net.parameters()]

    return a, grad_W

def set_options(options):
    '''Sets policy gradient options.'''
    global cuda, max_episode_len, max_len_penalty, FloatTensor, ZeroTensor, ByteTensor
    cuda = options.cuda
    max_episode_len = options.max_episode_len
    max_len_penalty = options.max_len_penalty
    FloatTensor = lambda x: torch.cuda.FloatTensor(x) if cuda else torch.FloatTensor(x)
    ZeroTensor = lambda *s: torch.cuda.FloatTensor(*s).zero_() if cuda else torch.zeros(*s)
    ByteTensor = lambda x: torch.cuda.ByteTensor(x) if cuda else torch.ByteTensor(x)

    # Transparently set number of threads based on environment variables
    num_threads = int(os.getenv('OMP_NUM_THREADS', 1))
    torch.set_num_threads(num_threads)

    # Monkey-patch options in policy_gradient.py
    policy_gradient.set_options(options)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs multi-agent policy gradient.')
    parser.add_argument('--game', choices=['gridworld', 'gridworld_3d'], required=True, help='A game to run')
    parser.add_argument('--cuda', default=False, action='store_true', help='Include to run on CUDA')
    parser.add_argument('--max_episode_len', default=float('inf'), type=float, help='Terminate episode early at this number of steps')
    parser.add_argument('--max_len_penalty', default=0, type=float, help='If episode is terminated early, add this to the last reward')
    parser.add_argument('--num_episodes', default=100000, type=int, help='Number of episodes to run in a round of training')
    parser.add_argument('--num_rounds', default=1, type=int, help='How many rounds of training to run')
    parser.add_argument('--policy_net_opt', default='rmsprop', choices=['rmsprop', 'rprop'], help='Optimizer for training the policy net')
    parser.add_argument('--td_update', type=int, help='k for a TD(k) update term for the policy and value nets; exclude for a Monte-Carlo update')
    parser.add_argument('--gamma', default=1, type=float, help='Global discount factor for Monte-Carlo and TD returns')
    args = parser.parse_args()
    set_options(args)

    if args.game == 'gridworld':
        import gridworld as game
        policy_net_layers = [2, 32, 9]
        value_net_layers = [2, 32, 1]
        game.set_options({'grid_y': 4, 'grid_x': 4})
    if args.game == 'gridworld_3d':
        import gridworld_3d as game
        policy_net_layers = [3, 64, 27]
        value_net_layers = [3, 32, 1]
        game.set_options({'grid_z': 4, 'grid_y': 4, 'grid_x': 4})

    for i in range(args.num_rounds):
        policy_net = build_policy_net(policy_net_layers)
        value_net = build_value_net(value_net_layers)

        # Init main Tensors first, so we don't have to allocate memory at runtime
        # TODO: Check again after https://github.com/pytorch/pytorch/issues/339
        #   Used in run_policy_net():
        a_size = policy_net_layers[2]
        #   Used in train_policy_net():
        W_step = [ZeroTensor(W.size()) for W in policy_net.parameters()]
        mean_square = [ZeroTensor(W.size()) for W in policy_net.parameters()]
        for W in mean_square: W += 1

        # Monkey-patch globals from policy_gradient.py
        policy_gradient.a_size = a_size
        policy_gradient.W_step = W_step
        policy_gradient.mean_square = mean_square

        avg_value_error, avg_return = 0.0, 0.0
        for num_episode in range(args.num_episodes):
            episode = run_episode(policy_net, gamma=args.gamma)
            value_error = train_value_net(value_net, episode, td=args.td_update, gamma=args.gamma)
            avg_value_error = 0.9 * avg_value_error + 0.1 * value_error
            avg_return = 0.9 * avg_return + 0.1 * episode[0].G
            print("{{'i': {}, 'num_episode': {}, 'episode_len': {}, 'episode_return': {}, 'avg_return': {}, 'avg_value_error': {}}},".format(i, num_episode, len(episode), episode[0].G, avg_return, avg_value_error))
            train_policy_net(policy_net, episode, val_baseline=value_net, td=args.td_update, gamma=args.gamma, opt=args.policy_net_opt)

