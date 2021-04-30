import numpy as np
import random
from collections import defaultdict

REWARD = -0.1
GAMMA = 0.7
DISCOUNT = 0.9 # discount 1 takes forever
MAX_ERROR = 10**(-3)

# Parameters for MC
EPISODES = 10
START = (3,2)
TERMINAL_STATES = [(3,0),(0,3),(3,2)]

ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # actions which can be D, L, U, R
NUM_ACTIONS = len(ACTIONS)

# Grid size based on amount of rows and columns
ROW = 4
COL = 4

# Grid with terminal states
U = [[0, 0, 0, 40],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [10, 0, 0, 0]]


def maze_grid(arr, policy=False):
    """[summary]

    Args:
        arr ([type]): [description]
        policy (bool, optional): [description]. Defaults to False.
    """
    res = ""
    for r in range(ROW):
        res += "|"
        for c in range(COL):
            if r == 0 and c == 3:
                val = "0"
            elif r == 3 and c == 0:
                val = "0"
            else:
                val = ['\u2193', '\u2190', '\u2191', '\u2192'][arr[r][c]]
            res += " " + val[:5].ljust(5) + " |"  # format
        res += "\n"
    print(res)


def get_utility(U, r, c, action):
    """This function gets the utility of the state reached by performing the
    given action from the given state

    Args:
        U ([type]): [description]
        r ([type]): [description]
        c ([type]): [description]
        action ([type]): [description]

    Returns:
        [type]: [description]
    """
    dr, dc = ACTIONS[action]
    newR = r + dr
    newC = c + dc
    start = (newR == 3 and newC == 2)
    if newR < 0 or newC < 0 or newR >= ROW or newC >= COL or start:
        return U[r][c]
    else:
        return U[newR][newC]


def calculate_utility(U, r, c, action, DISCOUNT):
    """This function calculates the utility of a state given
    an action

    Args:
        U ([type]): [description]
        r ([type]): [description]
        c ([type]): [description]
        action ([type]): [description]

    Returns:
        [type]: [description]
    """
    if r == 3 and c == 1:
        u = -2
    elif r == 1 and c == 2:
        u = -10
    elif r == 1 and c == 3:
        u = -10
    else:
        u = REWARD

    u += 0.1 * DISCOUNT * get_utility(U, r, c, (action-1)%4)
    u += GAMMA * DISCOUNT * get_utility(U, r, c, action)
    u += 0.1 * DISCOUNT * get_utility(U, r, c, (action+1)%4)

    return u


def policy_evaluation(policy, U, DISCOUNT):
    """Perform some simplified value iteration steps to get an approximation of
    the utilities

    Args:
        policy ([type]): [description]
        U ([type]): [description]

    Returns:
        [type]: [description]
    """
    while True:
        nextU = [[0, 0, 0, 40],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [10, 0, 0, 0]]
        error = 0
        for r in range(ROW):
            for c in range(COL):
                terminal_states = [(r == 0 and c == 3),
                                (r == 3 and c == 0),
                                (r == 3 and c == 1)]
                if any(terminal_states):
                    continue
                # simplified Bellman update
                nextU[r][c] = calculate_utility(U, r, c, policy[r][c]) # don't use max
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        if error < MAX_ERROR * (1-DISCOUNT) / DISCOUNT:
            break
    return U


def policy_iteration(policy, U, DISCOUNT):
    """[summary]

    Args:
        policy ([type]): [description]
        U ([type]): [description]

    Returns:
        [type]: [description]
    """
    print("During the policy iteration:\n")
    while True:
        U = policy_evaluation(policy, U, DISCOUNT)
        unchanged = True
        for r in range(ROW):
            for c in range(COL):
                terminal_states = [(r == 0 and c == 3),
                                   (r == 3 and c == 0),
                                   (r == 3 and c == 1)]
                if any(terminal_states):
                    continue
                maxAction, maxU = None, -float("inf")
                for action in range(NUM_ACTIONS):
                    u = calculate_utility(U, r, c, action, DISCOUNT)
                    if u > maxU:
                        maxAction, maxU = action, u
                if maxU > calculate_utility(U, r, c, policy[r][c], DISCOUNT):
                    policy[r][c] = maxAction
                    unchanged = False
        if unchanged:
            break
        maze_grid(policy)
    return policy


def create_random_policy(row, col, min, max):
    """Creates a random policy

    Args:
        row ([type]): [description]
        col ([type]): [description]
        min ([type]): [description]
        max ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.random.randint(min, max, size=(row, col))


# # Print the initial environment
# print("The initial random policy is:\n")
# policy = create_random_policy(ROW,COL,0,3)
# maze_grid(policy)

# # Policy iteration / optimal policy
# policy = policy_iteration(policy, U)
# # print(policy)

# # Print the optimal policy
# print("The optimal policy is:\n")
# maze_grid(policy)

# policy = create_random_policy(4,4,0,3)
# print(policy[START])
