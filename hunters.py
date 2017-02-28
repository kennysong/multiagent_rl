'''
    This is a multi-agent learning task, where k hunters (agents) are trying to
    catch m rabbits in an nxn grid.

    Hunters and rabbits are initialized randomly on the grid, with overlaps.
    An episode ends when all rabbits have been captured. Rabbits can have
    different movement patterns. There is a reward of -1 per time step (and
    optionally a +1 reward on capturing a rabbit).

    States are size 2*k+2*m flattened arrays of:
      concat(hunter positions, rabbit positions)
    Positions are of the form:
      [0, 0] = top-left, [n-1, n-1] = top-bottom, [-1, -1] = removed

    Actions are size 2*k flattened arrays of:
      concat(hunter 1 movement, hunter 2 movement, ..., hunter k movement)
    Movements are of the form:
      [0, 1] = right, [-1, 1] = up-right, [0, 0] = stay, etc
'''

import numpy as np

n = 6  # grid size
k = 2  # hunters
m = 2  # rabbits
num_agents = k

# None: rabbits do not move
# 'random': rabbits move one block randomly
# 'opposite': rabbits move opposite to the closest hunter
rabbit_action = None
# remove_hunter will remove the first hunter that captures a rabbit
remove_hunter = False
# capture_reward is the extra reward if a rabbit is captured
capture_reward = 0

def start_state():
    '''Returns a random initial state. The state vector is a flat array of:
        concat(hunter positions, rabbit positions).'''
    return np.random.randint(0, n, size=2*k+2*m)

def valid_state(s):
    '''Returns if the given state vector is valid.'''
    return s.shape == (2*k+2*m, ) and np.all([-1 <= e < n for e in s])

def valid_action(a):
    '''Returns if the given action vector is valid'''
    return a.shape == (2*k, ) and np.all([-1 <= e <= 1 for e in a])

def perform_action(s, a_indices):
    '''Performs an action given by a_indices in state s. Returns:
       (s_next, reward)'''
    # Validate inputs
    a = action_indices_to_coordinates(a_indices)
    assert valid_state(s)
    assert valid_action(a)

    # Calculate rabbit actions
    if rabbit_action is None:
        rabbit_a = np.zeros(2*m, dtype=np.int)
    elif rabbit_action == 'random':
        rabbit_a = np.random.randint(-1, 2, size=2*m)
    elif rabbit_action == 'opposite':
        rabbit_a = np.zeros(2*m, dtype=np.int)
        for i in range(0, 2*m, 2):
            rabbit_a[i:i+2] = opposite_direction(s, a, 2*k+i)
    else:
        raise ValueError('Invalid rabbit_action')

    # Get positions after hunter and rabbit actions
    a = np.concatenate((a, rabbit_a))
    positions = np.zeros(len(s), dtype=np.int)
    for i in range(len(s)):
        if s[i] == -1:
            positions[i] = s[i]
        elif 0 <= s[i] + a[i] < n:
            positions[i] = s[i] + a[i]
        else:
            positions[i] = s[i]

    # Remove rabbits (and optionally hunters) that overlap
    reward = -1
    hunter_pos, rabbit_pos = positions[:2*k], positions[2*k:]
    for i in range(0, len(hunter_pos), 2):
        hunter = hunter_pos[i:i+2]
        for j in range(0, len(rabbit_pos), 2):
            rabbit = rabbit_pos[j:j+2]
            if array_equal(hunter, rabbit) and hunter[0] != -1:
                # A rabbit has been captured
                rabbit_pos[j:j+2] = [-1, -1]
                reward += capture_reward
                if remove_hunter: hunter_pos[i:i+2] = [-1, -1]

    # Return (s_next, reward)
    s_next = np.concatenate((hunter_pos, rabbit_pos))
    return s_next, reward

def opposite_direction(s, a, i):
    '''Returns the direction the rabbit at s[i], s[i+1] should move to avoid
       the closest hunter (after hunters take action a).
    '''
    # Calculate hunter positions after a
    hunter_s = np.array(s[:2*k])
    for j in range(2*k):
        if hunter_s[j] == -1:
            continue
        elif 0 <= hunter_s[j] + a[j] < n:
            hunter_s[j] += a[j]

    # Find position of closest hunter
    rabbit = s[i:i+2]
    distance = float('inf')
    for j in range(0, 2*k, 2):
        d = np.linalg.norm(rabbit - s[j:j+2])
        if d < distance:
            closest_hunter = s[j:j+2]

    # Calculate opposite direction
    return np.sign(rabbit - closest_hunter)

def is_end(s):
    '''Given a state, return if the game should end.'''
    rabbit_pos = s[2*k:]
    return (rabbit_pos == -1).all()

def array_equal(a, b):
    '''Because np.array_equal() is too slow. Two-element arrays only.'''
    return a[0] == b[0] and a[1] == b[1]

def set_options(options):
    '''Set some game options, if given.'''
    global rabbit_action, remove_hunter, capture_reward
    rabbit_action = options.get('rabbit_action', rabbit_action)
    remove_hunter = options.get('remove_hunter', remove_hunter)
    capture_reward = options.get('capture_reward', capture_reward)

## Functions to convert action representations ##

action_index_to_coords = [
    np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]),
    np.array([0, -1]), np.array([0, 0]), np.array([0, 1]),
    np.array([1, -1]), np.array([1, 0]), np.array([1, 1])
]

def action_index_to_coordinates(index):
    '''Converts an action index 0 to 8 to an agent's action coordinates.'''
    assert 0 <= index <= 8
    return action_index_to_coords[index]

def action_indices_to_coordinates(a_indices):
    '''Converts a list of action indices to action coordinates.'''
    coords = [action_index_to_coordinates(i) for i in a_indices]
    return np.concatenate(coords)

def action_coordinates_to_index(coords):
    '''Converts an agent's action coordinates to an index 0 to 8.'''
    assert -1 <= coords[0] <= 1 and -1 <= coords[1] <= 1
    matches = [np.array_equal(coords, c) for c in action_index_to_coords]
    return matches.index(True)

## Functions to convert state representations ##

state_index_to_coords = [np.array((col, row)) for col in range(n)
                                              for row in range(n)] + \
                        [np.array((-1, -1))]

def state_index_to_coordinates(index):
    '''Converts a state index 0 to (n*n-1) to an agent's state coordinates.'''
    assert 0 <= index < n*n
    return state_index_to_coords[index]

def state_coordinates_to_index(coords):
    '''Converts an agent's state coordinates to an index 0 to (n*n-1).'''
    assert -1 <= coords[0] < n and -1 <= coords[1] < n
    matches = [np.array_equal(coords, c) for c in state_index_to_coords]
    return matches.index(True)

def state_coordinates_to_kmhot(state):
    '''Converts a coordinate state vector to a (k+m)-hot state vector.'''
    onehots = []
    for i in range(k+m):
        s = state[2*i:2*i+2]
        if np.array_equal(s, np.array([-1, -1])):
            # All zero vector corresponds to [-1, -1]
            onehots.append(np.zeros(n*n))
        else:
            index = state_coordinates_to_index(s)
            onehot = np.zeros(n*n)
            onehot[index] = 1
            onehots.append(onehot)
    return np.concatenate(onehots)

def state_kmhot_to_coordinates(kmhot):
    '''Converts a (k+m)-hot state vector to a coordinate state vector.'''
    coords = []
    for i in range(k+m):
        onehot = kmhot[i*(n*n):(i+1)*(n*n)]
        where = np.where(onehot == 1)
        if len(where[0]) == 0:
            # All zero vector corresponds to [-1, -1]
            coords.append(np.array([-1, -1]))
        else:
            index = where[0][0]
            state = state_index_to_coordinates(index)
            coords.append(state)
    return np.concatenate(coords)
