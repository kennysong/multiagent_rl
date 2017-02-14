'''
    This is a policy gradient implementation (REINFORCE with v(s) baseline)
    on the two-agent Gridworld Cliff environment.
'''

import gridworld
import numpy as np
import random
import torch

from torch.autograd import Variable

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
    result = value_net(Variable(torch.Tensor([state])))
    return result.data

value_net = build_value_network()
for i in range(1000):
    episode = run_episode()
    loss = train_value_network(value_net, episode)
    print(i, loss)
