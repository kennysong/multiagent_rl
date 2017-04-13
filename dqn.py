"""
Outline:
    run_episode(Q_net, eps=0.1):
        Run the episode
        Store (s_t, a_t, r_t, s_t+1) in buffer
        Return nothing

    train_Q_net(Q_net, Q_target_net, gamma, lr=1e-3, opt='RMSprop')
        Sample list of [(s_t, a_t, r_t, s_t+1)]
        Create batch of s_t, and of s_t+1
        Calculate y_t = r_t + gamma * max(Q_target_net(s_t+1))
        Calculate qvals = Q(s_t)[a_t]
        Calculate gradients g_w = Q(s_t)[a_t].backward()
        Call opt(Q_net_params, (y_t - qvals)g_w)
"""
import argparse
import numpy as np
import os
import random
import sys
import torch
import time

from namedlist import namedlist
from torch.autograd import Variable
from visualize import make_dot

# Define a EpisodeStep container for each step in an episode:
#   s, a is the state-action pair visited during that step
#   r is the reward received from that state-action pair
#   G is the discounted return received from that state-action pair
EpisodeStep = namedlist('EpisodeStep', 's a r G next_s', default=0)
SMALL = 1e-7

def onehot(index, size):
    l = [0 for i in range(size)]
    l[index] = 1
    return l

def masked_max(x, mask):
    """ Performs a masked max operation along the second dimension of x.

    Parameters:
        x: batch_size x d FloatTensor or variable to be maxed
        mask: batch_size x d FloatTensor or variable to mask the max

    Returns:
        y: batch_size x 1 FloatTensor with the max
    """

    offsets = x.min(1)[0]

    y = ((x - offsets.expand_as(x)) * mask).max(1)[0] + offsets

    return y


def build_Q_net(layers):
    class QNet(torch.nn.Module):
        def __init__(self, layers):
            super(QNet, self).__init__()
            assert len(layers) > 2, "Insufficient number of layes"
            self.layers = layers
            self.input_to_h = torch.nn.Linear(self.layers[0], self.layers[1])
            self.hidden_layers = [torch.nn.Linear(self.layers[l], self.layers[l+1]) for l in range(len(self.layers))[1:-2] ]
            self.hidden_to_out = torch.nn.Linear(self.layers[-2], self.layers[-1])
        def forward(self, x):
            z = self.input_to_h(x)
            h = torch.nn.ReLU()(z)
            for linear_layer in self.hidden_layers:
                z = linear_layer(h)
                h = torch.nn.ReLU()(z)
            Qvals = self.hidden_to_out(h)
            return Qvals
    q_net = QNet(layers)
    return q_net.cuda() if cuda else q_net

def train_q_net(Q_net, batch_size, gamma=0.99, target_network = None, episode=None):
    if target_network is None:
        target_network = Q_net
    if episode is not None:
        tuples_from_buffer = episode
        batch_size = len(episode)
    else:
        tuples_from_buffer = [replay_buffer[i] for i in np.random.randint(len(replay_buffer), size=(batch_size))]
    s_t_batch = Variable(FloatTensor([step.s for step in tuples_from_buffer]))
    s_t_next_batch = Variable(FloatTensor([step.next_s for step in tuples_from_buffer]))
    a_t_indexes = [step.a for step in tuples_from_buffer]
    qvals_all = Q_net(s_t_batch)
    
    qvals_t = Variable(ZeroTensor(batch_size))
    for i in range(batch_size):
        qvals_t[i] = qvals_all[i, a_t_indexes[i]]

    action_mask_batch = ZeroTensor(batch_size, a_size)
    for j, step in enumerate(tuples_from_buffer):
        action_mask_batch[j,:].copy_(torch.Tensor(game.filter_joint_actions(step.next_s)))

    qvals_target_all = target_network(s_t_next_batch)
    qvals_target_max = masked_max(qvals_target_all.data, action_mask_batch) # Note the detachment in qvals_target_all
    qvals_target_max.squeeze_()

    rewards = FloatTensor([step.r for step in tuples_from_buffer])
    targets = rewards + gamma * qvals_target_max
    
    errors = ((Variable(targets) - qvals_t)**2).mean()

    errors.backward()

    optimizer_Q_net.step()


def run_episode(Q_net, epsilon, gamma=1.0):
    '''Runs one episode of Gridworld Cliff to completion with a Q network,
       which is a MLP that maps states to state action pairs.

       Parameters:
       policy_net is our MLP policy network
       gamma is the discount factor for calculating returns
       epsilon factor for epsilon greedy

       Returns:
       [EpisodeStep(t=0), ..., EpisodeStep(t=T)]
    '''
    # Initialize state as player position
    state = game.start_state()
    episode = []

    # Run game until agent reaches the end
    while not game.is_end(state):
        # Let our agent decide that to do at this state
        input = Variable(FloatTensor(state).unsqueeze(0))
        Qvals = Q_net(input)
        
        action_mask = FloatTensor(game.filter_joint_actions(state)).unsqueeze(0)
        filtered_qvals = ((Qvals.data - Qvals.data.min() + SMALL) * action_mask)
        a = filtered_qvals.max(1)[1][0,0]
        if np.random.rand() < epsilon:
            a = torch.multinomial(action_mask, 1)[0,0]

        # Take that action, then the game gives us the next state and reward
        next_s, r = game.perform_joint_action(state, onehot(a, a_size))

        # Record state, action, grad_W, reward
        episode.append(EpisodeStep(s=state, a=a, r=r, next_s=next_s))
        replay_buffer.append(EpisodeStep(s=state, a=a, r=r, next_s=next_s))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs multi-agent policy gradient.')
    parser.add_argument('--game', choices=['gridworld', 'gridworld_3d'], required=True, help='A game to run')
    parser.add_argument('--cuda', default=False, action='store_true', help='Include to run on CUDA')
    parser.add_argument('--max_episode_len', default=float('inf'), type=float, help='Terminate episode early at this number of steps')
    parser.add_argument('--max_len_penalty', default=0, type=float, help='If episode is terminated early, add this to the last reward')
    parser.add_argument('--num_episodes', default=100000, type=int, help='Number of episodes to run in a round of training')
    parser.add_argument('--num_rounds', default=1, type=int, help='How many rounds of training to run')
    parser.add_argument('--opt', default='rmsprop', choices=['rmsprop', 'rprop'], help='Optimizer for training')
    parser.add_argument('--gamma', default=.99, type=float, help='Global discount factor for Monte-Carlo and TD returns')
    parser.add_argument('--buffer_size', default=5000, type=int, help='Capacity of the replay buffer')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of the replay buffer')
    parser.add_argument('--epsilon', default=0.1, type=float, help='Epsilon parameter for epsilon greedy')
    args = parser.parse_args()
    print(args)
    
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
        Q_net_layers = [2, 32, 9]
        game.set_options({'grid_y': 4, 'grid_x': 4})
    if args.game == 'gridworld_3d':
        import gridworld_3d as game
        Q_net_layers = [3, 64, 27]
        game.set_options({'grid_z': 4, 'grid_y': 4, 'grid_x': 4})

    for i in range(args.num_rounds):
        Q_net = build_Q_net(Q_net_layers)
        optimizer_Q_net = torch.optim.RMSprop(Q_net.parameters(), lr=1e-3, eps=1e-5)

        # Init main Tensors first, so we don't have to allocate memory at runtime
        # TODO: Check again after https://github.com/pytorch/pytorch/issues/339
        #   Used in run_policy_net():
        a_size = Q_net_layers[-1]
        #   Used in train_policy_net():

        # Monkey-patch globals from policy_gradient.py
        replay_buffer = []
        avg_return = 0.0
        for num_episode in range(args.num_episodes):
            episode = run_episode(Q_net, args.epsilon, gamma=args.gamma)
            args.epsilon *= 0.9999
            if len(replay_buffer) > args.buffer_size:
                replay_buffer = replay_buffer[args.buffer_size/2:]
            avg_return = 0.9 * avg_return + 0.1 * episode[0].G
            print("{{'i': {}, 'num_episode': {}, 'episode_len': {}, 'episode_return': {}, 'avg_return': {}, 'epsilon': {}}},".format(i, num_episode, len(episode), episode[0].G, avg_return, args.epsilon))
            if args.batch_size < len(replay_buffer):
                train_q_net(Q_net, batch_size=args.batch_size, gamma=args.gamma)

