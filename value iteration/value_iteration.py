import numpy as np
import random

REWARD = -0.1
GAMMA = 0.7
DISCOUNT = 0.99

ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # actions which can be D, L, U, R
N_ACTIONS = len(ACTIONS)

# Grid size based on amount of rows and columns
ROW = 4
COL = 4

# Grid with terminal states
u = [[0, 0, 0, 40],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [10, 0, 0, 0]]


def maze_grid(arr, policy=False):
    """This function initializes the maze grid with all of the
    given rewards per state and prints it out nicely.
    The grid has 4 rows and 4 columns which is set in the cell above.
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
                if policy:
                    # arrow symbols
                    val = ['\u2193', '\u2190', '\u2191', '\u2192'][arr[r][c]]
                else:
                    val = str(arr[r][c])
            res += " " + val[:5].ljust(5) + " |"
        res += "\n"
    print(res)


def get_utility(rewards, r, c, action):
    """This function gets the utility of the state reached
    by performing the given action from the given state
    """
    dr, dc = ACTIONS[action]
    newR = r + dr
    newC = c + dc
    start = (newR == 3 and newC == 2)
    if newR < 0 or newC < 0 or newR >= ROW or newC >= COL or start:
        return rewards[r][c]
    else:
        return rewards[newR][newC]


def calculate_utility(rewards, r, c, action):
    """This function calculates the utility of a state given
    an action
    """
    if r == 3 and c == 1:
        u = -2
    elif r == 1 and c == 2:
        u = -10
    elif r == 1 and c == 3:
        u = -10
    else:
        u = REWARD

    u += 0.1 * DISCOUNT * get_utility(rewards, r, c, (action-1)%4)
    u += GAMMA * DISCOUNT * get_utility(rewards, r, c, action)
    u += 0.1 * DISCOUNT * get_utility(rewards, r, c, (action+1)%4)

    return u

def value_iteration(rewards):
    """This function initializes the maze grid with all of the
    given rewards per state and prints it out nicely.
    The grid has 4 rows and 4 columns which is set in the cell above.
    """
    print("During the value iteration:\n")
    while True:
        next_utility = [[0, 0, 0, 40],
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
                next_utility[r][c] = max([calculate_utility(rewards, r, c, action) for action in range(N_ACTIONS)]) # Bellman update
                error = max(error, abs(next_utility[r][c]-rewards[r][c]))
        rewards = next_utility
        maze_grid(rewards)
        if error < ((1-GAMMA) / GAMMA):
            break
    return rewards


def optimal_policy(rewards):
    """This function gets the optimal policy from U
    """
    policy = [[-1,-1,-1,-1],
              [-1,-1,-10,-10],
              [-1,-1,-1,-1],
              [-1,-2,-1,-1]]

    for r in range(ROW):
        for c in range(COL):
            terminal_states = [(r == 0 and c == 3),
                               (r == 3 and c == 0),
                               (r == 3 and c == 1)]
            if any(terminal_states):
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(N_ACTIONS):
                u = calculate_utility(rewards, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy


def create_random_policy(row, col, min, max):
    """Creates a random policy.
    """
    pol = np.random.randint(min, max, size=(row, col))
    pol_grid = maze_grid(pol, True)
    return pol, pol_grid


# Print the initial environment
print("The initial maze grid is:\n")
maze_grid(u)

# Value iteration
rewards = value_iteration(u)
# print(rewards)

# Get the optimal policy from U and print it
policy = optimal_policy(rewards)
print("The optimal policy is:\n")
maze_grid(policy, True)
