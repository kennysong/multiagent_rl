'''
    This is a multi-agent learning task, where k hunters (agents) are trying to
    catch m rabbits in an nxn grid.

    Hunters and rabbits are initialized randomly on the grid, with overlaps.
    An episode ends when all rabbits have been captured. Rabbits can have
    different movement patterns. There is a reward of -1 per time step (and
    optionally a +1 reward on capturing a rabbit).

    States are size 3*k+3*m flattened arrays of:
      concat(hunter positions, rabbit positions)
    Positions are of the form:
      [in-game, y-position, x-position], so
      [1, 0, 0] = top-left, [1, 0, n-1] = top-right, [0, -1, -1] = removed

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
# timestep_reward is the reward given at each time-step
timestep_reward = -1

def start_state():
    '''Returns a random initial state. The state vector is a flat array of:
        concat(hunter positions, rabbit positions).'''
    start = np.random.randint(0, n, size=3*k+3*m)
    start[::3] = 1
    return start

def valid_state(s):
    '''Returns if the given state vector is valid.'''
    return s.shape == (3*k+3*m, ) and \
           np.all([-1 <= e < n for e in s]) and \
           np.all([e in (0, 1) for e in s[::3]])

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
    for i in range(0, len(s), 3):
        if s[i] == 0:
            positions[i:i+3] = [0, -1, -1]
        else:
            positions[i] = 1
            sa = s[i+1:i+3] + a[i-(i/3):i-(i/3)+2]
            positions[i+1:i+3] = np.clip(sa, 0, n-1)

    # Remove rabbits (and optionally hunters) that overlap
    reward = timestep_reward
    hunter_pos, rabbit_pos = positions[:3*k], positions[3*k:]
    for i in range(0, len(hunter_pos), 3):
        hunter = hunter_pos[i:i+3]
        for j in range(0, len(rabbit_pos), 3):
            rabbit = rabbit_pos[j:j+3]
            if hunter[0] == 1 and rabbit[0] == 1 and array_equal(hunter, rabbit):
                # A rabbit has been captured
                rabbit_pos[j:j+3] = [0, -1, -1]
                reward += capture_reward
                if remove_hunter: hunter_pos[i:i+3] = [0, -1, -1]

    # Return (s_next, reward)
    s_next = np.concatenate((hunter_pos, rabbit_pos))
    return s_next, reward

def perform_joint_action(s, joint_a):
    '''Performs an action given by joint_a in state s. Returns:
       (s_next, reward)'''
    a_indices = joint_action_to_indices(joint_a)
    return perform_action(s, a_indices)

def filter_actions(state, agent_no):
    '''Filter the actions available for an agent in a given state. Returns a
       bitmap of available actions. Hunters out of the game can only choose
       the "stay" action.
       E.g. an agent in a corner is not allowed to move into a wall.'''
    avail_a = np.ones(9, dtype=int)
    hunter_pos = state[3*agent_no + 1:3*agent_no + 3]

    # Hunter is out of the game, can only stay
    if state[3*agent_no] == 0:
        avail_a = [0] * 9
        avail_a[4] = 1
        return avail_a

    # Hunter is still in the game, check all possible actions
    for i in range(9):
        # Check if action moves us off the grid
        a = agent_action_space[i]
        sa = hunter_pos + a
        if (sa[0] < 0 or sa[0] >= n) or (sa[1] < 0 or sa[1] >= n):
            avail_a[i] = 0
    return avail_a

def filter_joint_actions(state):
    '''Filter the actions available in a given state. Returns a bitmask of
       available actions. Hunters out of the game can only choose
       the "stay" action.
       E.g. an agent in a corner is not allowed to move into a wall.'''

    def _select_idx(actions, agent):
        '''Returns all indexes that involve a specific action for one agent.'''
        idx = np.zeros(9**k, dtype=bool)
        for act in actions:
            for s in range(act*(9**agent), 9**k, 9**(agent+1)):  # Magic
                idx[s:s+9**agent] = 1
        return idx

    # Start with a ones vector and start invalidating batches of actions
    avail_a = np.ones(9**k, dtype=int)

    # If a hunter is out, invalidate actions except stay (index 4)
    for agent in range(k):
        if state[3*agent] == 0:  # Status bit == 0
            idx = ~_select_idx([4], agent)
            avail_a[idx] = 0

    # If a hunter is on a border, invalidate actions that move off the grid
    for agent in range(k):
        if state[3*agent] == 0: continue  # Only look at hunters in the game
        pos = state[3*agent+1:3*agent+3]
        if pos[0] == 0:  # Against top wall
            idx = _select_idx([0, 1, 2], agent)
            avail_a[idx] = 0
        elif pos[0] == n-1:  # Against bottom wall
            idx = _select_idx([6, 7, 8], agent)
            avail_a[idx] = 0

        if pos[1] == 0:  # Against left wall
            idx = _select_idx([0, 3, 6], agent)
            avail_a[idx] = 0
        elif pos[1] == n-1:  # Against right wall
            idx = _select_idx([2, 5, 8], agent)
            avail_a[idx] = 0

    return avail_a

# def filter_joint_actions_test(state):
#     '''This is just a brute force validator for filter_joint_actions().'''
#     joint_a = [0] * (9**k)
#     avail_a = [1] * (9**k)

#     # Check all possible actions (a lot!)
#     for joint_a_num in range(9**k):
#         # Convert the joint action into action indices
#         if joint_a_num > 0: joint_a[joint_a_num - 1] = 0
#         joint_a[joint_a_num] = 1
#         a_indices = joint_action_to_indices(joint_a)

#         # If all agents have valid action indices, this is a valid joint action
#         for agent_no, a_index in enumerate(a_indices):
#             avail_agent_a = filter_actions(state, agent_no)
#             if avail_agent_a[a_index] == 0:
#                 avail_a[joint_a_num] = 0
#                 break
#     return np.array(avail_a)

def is_end(s):
    '''Given a state, return if the game should end.'''
    rabbit_status = s[3*k::3]
    return (rabbit_status == 0).all()

def array_equal(a, b):
    '''Because np.array_equal() is too slow. Three-element arrays only.'''
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

def set_options(options):
    '''Set some game options, if given.'''
    global rabbit_action, remove_hunter, timestep_reward, capture_reward, n, k, m, num_agents
    rabbit_action = options.get('rabbit_action', rabbit_action)
    remove_hunter = options.get('remove_hunter', remove_hunter)
    timestep_reward = options.get('timestep_reward', timestep_reward)
    capture_reward = options.get('capture_reward', capture_reward)
    n = options.get('n', n)
    k = options.get('k', k)
    m = options.get('m', m)
    num_agents = k
    print(options)

## Functions to convert action representations ##

agent_action_space = [
    np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]),
    np.array([0, -1]), np.array([0, 0]), np.array([0, 1]),
    np.array([1, -1]), np.array([1, 0]), np.array([1, 1])
]

def action_indices_to_coordinates(a_indices):
    '''Converts a list of action indices to action coordinates.'''
    coords = [agent_action_space[i] for i in a_indices]
    return np.concatenate(coords)

def joint_action_to_indices(joint_a):
    '''Convert a joint action into action indices. We use the transformation:
                    action for n'th hunter = (J//A^n) % A
       where J is the joint action number
             A is the number of actions for each agent
             n is the hunter number (starting from 0)

       Intuitively, the
         0th hunter will cycle through actions [0, A) on every +1 joint action number
         1st hunter will cycle through actions [0, A) on every +A joint action number
         2nd hunter will cycle through actions [0, A) on every +A^2 joint action number
         nth hunter will cycle through actions [0, A) on every +A^n joint action number
       (This is also base A, in reverse digit order.)
    '''
    a_indices = [None] * k
    for hunter in range(k):
        a_indices[hunter] = (joint_a // 9**hunter) % 9
    return a_indices

def opposite_direction(s, a, i):
    '''Returns the direction the rabbit at s[i], s[i+1] should move to avoid
       the closest hunter (after hunters take action a).
    '''
    raise NotImplementedError('TODO: allow rabbits to move in opposite direction')

    # # Calculate hunter positions after a
    # hunter_s = np.array(s[:2*k])
    # for j in range(2*k):
    #     if hunter_s[j] == -1:
    #         continue
    #     elif 0 <= hunter_s[j] + a[j] < n:
    #         hunter_s[j] += a[j]

    # # Find position of closest hunter
    # rabbit = s[i:i+2]
    # distance = float('inf')
    # for j in range(0, 2*k, 2):
    #     d = np.linalg.norm(rabbit - s[j:j+2])
    #     if d < distance:
    #         closest_hunter = s[j:j+2]

    # # Calculate opposite direction
    # return np.sign(rabbit - closest_hunter)
