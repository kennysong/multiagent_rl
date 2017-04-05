import numpy as np
import torch

from namedlist import namedlist
from torch.autograd import Variable

EpisodeStep = namedlist('EpisodeStep', 's a grad_W r G', default=0)

ByteTensor = lambda x: torch.ByteTensor(x)

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

def build_value_net(layers):
    value_net = torch.nn.Sequential(
                  torch.nn.Linear(layers[0], layers[1]),
                  torch.nn.Tanh(),
                  torch.nn.Linear(layers[1], layers[2]))
    return value_net

def filter_actions(state, agent_no):
    '''Filter the actions available for an agent in a given state. Returns a
       bitmap of available states.
       E.g. an agent in a corner is not allowed to move into a wall.'''
    actions = [1, 1, 1]
    # Vertical agent
    if agent_no == 0:
        if state[0] == 0: actions[0] = 0
        elif state[0] == grid_y-1: actions[2] = 0
    # Horizontal agent
    elif agent_no == 1:
        if state[1] == 0: actions[0] = 0
        elif state[1] == grid_x-1: actions[2] = 0
    return actions

def print_episode(episode_file):
    '''Prints episode states and actions.'''
    # Load serialized episode data
    episode = torch.load(episode_file, map_location=lambda storage, loc: storage)
    states = [step.s for step in episode]
    actions = [step.a for step in episode]

    print(states)
    print(actions)

def print_policy(policy_file):
    '''Prints a policy, i.e. probabilities of all actions from each state.'''
    # Load serialized policy parameters
    policy_weights = torch.load(policy_file, map_location=lambda storage, loc: storage)
    policy = PolicyNet([5, 32, 3])
    policy.load_state_dict(policy_weights)
    softmax = torch.nn.Softmax()

    # Get action probabilities from each state
    probs = []
    for state in state_space:
        prob = {}
        for a_v in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            # Forward step in LSTM for vertical agent
            x_0 = Variable(torch.Tensor(np.append(np.zeros(3), state).reshape(1, 5)))
            h_0, c_0 = Variable(torch.Tensor(np.zeros((1, 32)))), Variable(torch.Tensor(np.zeros((1, 32))))
            o_v, h_1, c_1 = policy(x_0, h_0, c_0)
            action_mask = ByteTensor(filter_actions(state, 0))
            filt_o_v = o_v[action_mask].resize(1, action_mask.sum())
            dist_v = softmax(filt_o_v)

            # Get full distribution for printing
            full_dist_v = [0, 0, 0]
            j = 0
            for i in range(len(action_mask)):
                if action_mask[i] == 1:
                    full_dist_v[i] = dist_v.data[0][j]
                    j += 1

            # Forward step in LSTM for horizontal agent 
            x_1 = Variable(torch.Tensor(np.append(a_v, state).reshape(1, 5)))
            o_h, _, _ = policy(x_1, h_1, c_1)
            action_mask = ByteTensor(filter_actions(state, 1))
            filt_o_h = o_h[action_mask].resize(1, action_mask.sum())
            dist_h = softmax(filt_o_h)

            # Get full distribution for printing
            full_dist_h = [0, 0, 0]
            j = 0
            for i in range(len(action_mask)):
                if action_mask[i] == 1:
                    full_dist_h[i] = dist_h.data[0][j]
                    j += 1

            # Calculate action joint probabilities
            if a_v == [1, 0, 0]:  # a_v is up
                prob['upleft'] = full_dist_v[0] * full_dist_h[0]
                prob['up'] = full_dist_v[0] * full_dist_h[1]
                prob['upright'] = full_dist_v[0] * full_dist_h[2]
            elif a_v == [0, 1, 0]:  # a_v is stay
                prob['left'] = full_dist_v[1] * full_dist_h[0]
                prob['stay'] = full_dist_v[1] * full_dist_h[1]
                prob['right'] = full_dist_v[1] * full_dist_h[2]
            elif a_v == [0, 0, 1]:  # a_v is down
                prob['downleft'] = full_dist_v[2] * full_dist_h[0]
                prob['down'] = full_dist_v[2] * full_dist_h[1]
                prob['downright'] = full_dist_v[2] * full_dist_h[2]
        probs.append(prob)

    # Print action probabilities as a table
    print('          upleft  up      upright  left    stay    right   downleft  down    downright')
    for state, prob in zip(state_space, probs):
        print('{:<9} {:1.4f}  {:1.4f}  {:1.4f}   {:1.4f}  {:1.4f}  {:1.4f}  {:1.4f}    {:1.4f}  {:1.4f}'.format(list(state),
                                             prob['upleft'],
                                             prob['up'],
                                             prob['upright'],
                                             prob['left'],
                                             prob['stay'],
                                             prob['right'],
                                             prob['downleft'],
                                             prob['down'],
                                             prob['downright']
                                             ))

def print_value_net(value_net_file):
    '''Prints a value network, i.e. values of all states.'''
    # Load serialized value net parameters
    value_net_params = torch.load(value_net_file, map_location=lambda storage, loc: storage)
    value_net = build_value_net([2, 32, 1])
    value_net.load_state_dict(value_net_params)

    # Print header of table
    header = '      '
    for col in range(grid_x):
        header += str(col) + '       '
    print(header)

    # Print values in a table
    for row in range(grid_y):
        row_str = str(row) + '  '
        for col in range(grid_x):
            state = Variable(torch.Tensor([[row, col]]))
            value = value_net(state).data[0][0]
            row_str += '{:1.3f}  '.format(value)
        print(row_str)

grid_y, grid_x = 4, 4 
state_space = [np.array((i, j)) for i in range(grid_y) for j in range(grid_x)]
print('Episode')
print_episode('4x4_episode')
print('\nPolicy for 4x4 grid at local optimum')
print_policy('4x4_policy')
print('\nValue Net for 4x4 grid at local optimum')
print_value_net('4x4_value')
