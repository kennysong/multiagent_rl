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

from namedlist import namedlist
from torch.autograd import Variable

# Define a EpisodeStep container for each step in an episode:
#   s, a is the state-action pair visited during that step
#   r is the reward received from that state-action pair
#   G is the discounted return received from that state-action pair
EpisodeStep = namedlist('EpisodeStep', 's a r G', default=0)
SMALL = 1e-7

def run_episode(policy_net, gamma=1.0):
    '''Runs one episode of Gridworld Cliff to completion with a policy network,
       which is a MLP that maps states to joint action probabilities.

       Parameters:
           policy_net: MLP policy network
           gamma: discount factor for calculating returns

       Returns:
           [EpisodeStep(t=0), ..., EpisodeStep(t=T)]
    '''
    # Initialize state as player position
    state = game.start_state()
    episode = []

    # Run game until agent reaches the end
    while not game.is_end(state):
        # Let our agent decide that to do at this state
        a = run_policy_net(policy_net, state)

        # Take that action, then the game gives us the next state and reward
        next_s, r = game.perform_joint_action(state, a)

        # Record state, action, reward
        episode.append(EpisodeStep(s=state, a=a, r=r))
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
       values. It has one hidden layer with tanh activations.
    '''
    value_net = torch.nn.Sequential(
                  torch.nn.Linear(layers[0], layers[1]),
                  torch.nn.Tanh(),
                  torch.nn.Linear(layers[1], layers[2]))
    value_net.layers = layers

    return value_net.cuda() if cuda else value_net

def train_value_net(value_net, episode, td=None, gamma=1.0):
    '''Trains an MLP value function approximator based on the output of one
       episode, i.e. first-visit Monte-Carlo policy evaluation. The value
       network will map states to scalar values.

       Warning: currently only works for integer-vector states!

       Parameters:
           value_net: value network to be trained
           episode: list of EpisodeStep's
           td: k for a TD(k) return, td=None for a Monte-Carlo return
           gamma: discount term used for TD(k) returns

       Returns:
           The scalar loss of the newly trained value network.
    '''
    # Pre-compute values, if being used
    if td is not None:
        values = [run_value_net(value_net, step.s) for step in episode]

    # Calculate return from the first visit to each state
    visited_states = set()
    states, returns = [], []
    for t in range(len(episode)):
        s, G = episode[t].s, episode[t].G
        str_s = s.astype(int).tostring()  # Fastest hashable state representation
        if str_s not in visited_states:
            visited_states.add(str_s)
            states.append(s)

            # Monte-Carlo return
            if td is None:
                returns.append(G)
            # TD return
            elif td >= 0:
                t_end = t + td + 1  # TD requires we look forward until t_end
                if t_end < len(episode):
                    r = sum([gamma**(j-t)*episode[j].r for j in range(t, t_end)]) + values[t_end]
                else:
                    r = sum([gamma**(j-t)*episode[j].r for j in range(t, len(episode))])
                returns.append(r)
    states = Variable(FloatTensor(states))
    returns = Variable(FloatTensor(returns))

    # Train the value network on states, returns
    optimizer_value_net.zero_grad()
    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(value_net(states), returns)
    loss.backward()
    optimizer_value_net.step()

    return loss.data[0]

def run_value_net(value_net, state):
    '''Wrapper function to feed one state into the given value network and
       return the value as a scalar.'''
    result = value_net(Variable(FloatTensor([state])))
    return result.data[0][0]

def build_policy_net(layers):
    '''Builds an MLP policy network, which maps states to action vectors.

       More precisely, the input into the MLP will be the state vector. The
       output of the MLP will be a vector that gives unnormalized probabilities
       of each joint action (one-hot vector of a combination of agents'
       actions). Softmax is applied afterwards, see run_policy_net().
    '''
    return build_value_net(layers)

def masked_softmax(logits, mask):
    """
    Parameters:
        logits: Variable of size [batch_size, d]
        mask: FloatTensor of size [batch_size, d]

    Returns:
        probs: row-wise masked softmax of the logits
    """
    # Based on numerically stable softmax, see:
    # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    # b must be the max over the unmasked logits
    inv_mask = ByteTensor(1 - mask.numpy().astype(int))
    inf_logits = logits.masked_fill(Variable(inv_mask), float('-inf'))
    b = torch.max(inf_logits, 1)[0].expand_as(inf_logits)

    # Calculate softmax; masked elements may explode, but are forced to 0
    scores = torch.exp(logits - b).masked_fill(Variable(inv_mask), 0)
    total_scores = torch.sum(scores, 1).expand_as(scores)
    probs = scores / total_scores
    return probs

def run_policy_net(policy_net, state):
    '''Wrapper function to feed a given state into the given policy network and
       return an action vector, as well as parameter gradients.

       Parameters:
           policy_net: LSTM that given (x_n, h_n, c_n) returns o_nn, h_nn, c_nn
           state: state of the MDP

       Returns:
           a_index: joint action index
    '''
    # Do a forward step through policy_net, filter actions, and softmax it
    x = Variable(FloatTensor([state]))
    o = policy_net(x)
    action_mask = FloatTensor(game.filter_joint_actions(state)).unsqueeze(0)
    dist = masked_softmax(o, action_mask)

    # Sample an available action from dist
    a_index = torch.multinomial(dist.data,1)[0,0]

    return a_index

def train_policy_net(policy_net, episode, val_baseline, td=None, gamma=1.0,
                     lr=3*1e-3):
    '''Update the policy network parameters with the REINFORCE algorithm.
       For each parameter W of the policy network, for each time-step t in the
       episode, make the update:
         W += alpha * [grad_W(LSTM(a_t | s_t)) * (G_t - baseline(s_t))]
            = alpha * [grad_W(sum(log(p))) * (G_t - baseline(s_t))]
       for all time steps in the episode.

       (Notes: The sum is over the number of agents, each with an associated p
               In practice, the time-steps are summed into one gradient update)

       Parameters:
           policy_net: LSTM policy network
           episode: list of EpisodeStep's
           val_baseline: value network used as the baseline term
           td: k for a TD(k) estimate of G_t, td=None for a Monte-Carlo G_t
           gamma: the discount term used for a TD(k) gradient term
    '''
    if td is not None:
        raise NotImplementedError('TD returns not implemented for the baseline!')

    # Pre-compute baselines
    values = [run_value_net(val_baseline, step.s) for step in episode]
    values = Variable(FloatTensor(np.asarray(values)))

    # Prepare for one forward pass, with the batch containing the entire episode
    a_size = policy_net.layers[2]
    s_size = len(episode[0].s)
    policy_net.zero_grad()

    # Fill input_batch with state for each agent, for each time-step
    input_batch = ZeroTensor(len(episode), s_size)
    for j, step in enumerate(episode):
        input_batch[j].copy_(torch.Tensor(step.s))
    input_batch = Variable(input_batch)

    # Fill action_mask_batch with action masks for each state
    action_mask_batch = ZeroTensor(len(episode), a_size)
    for j, step in enumerate(episode):
        action_mask_batch[j].copy_(torch.Tensor(game.filter_joint_actions(step.s)))

    # Do a forward pass, and fill sum_log_probs with sum(log(p)) for each time-step
    sum_log_probs = Variable(ZeroTensor(len(episode)))
    o = policy_net(input_batch)
    dist = masked_softmax(o, action_mask_batch)
    for j, step in enumerate(episode):
        sum_log_probs[j] = sum_log_probs[j] + torch.log(dist[j, step.a])

    # Do the backward pass to get grad_W(sum(log(p))) * (G_t - baseline(s_t))
    returns = Variable(FloatTensor(np.asarray([step.G for step in episode])))
    neg_performance = (sum_log_probs * (values - returns)).sum()
    neg_performance.backward()

    # Clip gradients to [-1, 1]
    for W in policy_net.parameters():
        W.grad.data.clamp_(-1,1)

    # Do a step of RMSProp
    optimizer_policy_net.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs multi-agent policy gradient.')
    parser.add_argument('--game', choices=['gridworld', 'gridworld_3d', 'hunters'], required=True, help='A game to run')
    parser.add_argument('--cuda', default=False, action='store_true', help='Include to run on CUDA')
    parser.add_argument('--max_episode_len', default=float('inf'), type=float, help='Terminate episode early at this number of steps')
    parser.add_argument('--max_len_penalty', default=0, type=float, help='If episode is terminated early, add this to the last reward')
    parser.add_argument('--num_episodes', default=100000, type=int, help='Number of episodes to run in a round of training')
    parser.add_argument('--num_rounds', default=1, type=int, help='How many rounds of training to run')
    parser.add_argument('--td_update', type=int, help='k for a TD(k) update term for the policy and value nets; exclude for a Monte-Carlo update')
    parser.add_argument('--gamma', default=1, type=float, help='Global discount factor for Monte-Carlo and TD returns')
    parser.add_argument('--save_policy', type=str, help='Save the trained policy under this filename')
    args = parser.parse_args()
    print(args)

    # Sets options for PG
    cuda = args.cuda
    max_episode_len = args.max_episode_len
    max_len_penalty = args.max_len_penalty
    if cuda: print('Running policy gradient on GPU.')

    # Transparently set number of threads based on environment variables
    num_threads = int(os.getenv('OMP_NUM_THREADS', 1))
    torch.set_num_threads(num_threads)

    # Define wrappers for Tensors
    FloatTensor = lambda x: torch.cuda.FloatTensor(x) if cuda else torch.FloatTensor(x)
    ZeroTensor = lambda *s: torch.cuda.FloatTensor(*s).zero_() if cuda else torch.zeros(*s)
    ByteTensor = lambda x: torch.cuda.ByteTensor(x) if cuda else torch.ByteTensor(x)

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
    elif args.game == 'hunters':
        # Note: Not sure how many hidden layers to give the policy net
        import hunters as game
        k, m = 3, 3
        if k == 1 or k == 2:
            policy_net_layers = [3*(k+m), 128, 9**k]
        elif k == 3:
            policy_net_layers = [3*(k+m), 1024, 9**k]
        elif k == 4:
            policy_net_layers = [3*(k+m), 8192, 9**k]
        value_net_layers = [3*(k+m), 64, 1]
        game.set_options({'rabbit_action': None, 'remove_hunter': True,
                          'timestep_reward': 0, 'capture_reward': 1,
                          'k': k, 'm': m})

    for i in range(args.num_rounds):
        policy_net = build_policy_net(policy_net_layers)
        value_net = build_value_net(value_net_layers)
        optimizer_policy_net = torch.optim.RMSprop(policy_net.parameters(), lr=1e-3, eps=1e-5)
        optimizer_value_net = torch.optim.RMSprop(value_net.parameters(), lr=1e-3, eps=1e-5)

        # RMSProp variables for policy net
        mean_square = [ZeroTensor(W.size()) for W in policy_net.parameters()]
        for W in mean_square: W += 1

        avg_value_error, avg_return = 0.0, 0.0
        for num_episode in range(args.num_episodes):
            episode = run_episode(policy_net, gamma=args.gamma)
            value_error = train_value_net(value_net, episode, td=args.td_update, gamma=args.gamma)
            avg_value_error = 0.9 * avg_value_error + 0.1 * value_error
            avg_return = 0.9 * avg_return + 0.1 * episode[0].G
            train_policy_net(policy_net, episode, value_net, td=args.td_update, gamma=args.gamma)
            print("{{'i': {}, 'num_episode': {}, 'episode_len': {}, 'episode_return': {}, 'avg_return': {}, 'avg_value_error': {}}},".format(i, num_episode, len(episode), episode[0].G, avg_return, avg_value_error))

        if args.save_policy is not None:
            if args.num_rounds > 1:
                torch.save(policy_net.state_dict(), args.save_policy + str(i))
            else:
                torch.save(policy_net.state_dict(), args.save_policy)
            print('Policy saved to ' + args.save_policy)
