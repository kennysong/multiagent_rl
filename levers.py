'''
    This is a multi-agent learning task, where m randomly selected agents (out
    of a pool of N agents) must pull m levers.

    The episode consists of only one time-step, and the reward is the number of
    distinct levers pulled.

    States are N-D binary vectors with m 1's that denote the selected agents.

    Actions are N-D vectors of integers in [0, m-1]. The i'th agent pulls the
      lever at index action[i]. Only selected agents in the state can actually
      pull a lever.
'''

import numpy as np

N = 10  # total agents
m = 5  # levers

def initial_state():
    '''Returns a random initial state.'''
    rand_indices = np.random.choice(range(N), size=m, replace=False)
    state = np.zeros(N, dtype=np.int)
    state[rand_indices] = 1
    return state

def valid_state(s):
    '''Returns if the given state vector is valid.'''
    return (s.sum() == m) and np.all([e in (0, 1) for e in s])

def valid_action(a):
    '''Returns if the given action vector is valid.'''
    return np.all([0 <= e < N for e in a])

def perform_action(s, a):
    '''Performs action a in state s. Returns reward.'''
    # Validate inputs
    assert valid_state(s)
    assert valid_action(a)

    # Return the number of distinct levers pulled
    levers = a[s == 1]
    return len(set(levers))
