import numpy as np

REWARD = -0.1
GAMMA = 0.7

ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # actions which can be D, L, U, R
NUM_ACTIONS = len(ACTIONS)

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
            elif r == 3 and c == 2:
                val += " S"
            else:
                if policy:
                    val = ['\u2193', '\u2190', '\u2191', '\u2192'][arr[r][c]] # arrow symbols
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
    if newR < 0 or newC < 0 or newR >= ROW or newC >= COL or (newR == 3 and newC == 2):
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
                
    u += 0.1 * GAMMA * get_utility(rewards, r, c, (action-1)%4)
    u += 0.7 * GAMMA * get_utility(rewards, r, c, action)
    u += 0.1 * GAMMA * get_utility(rewards, r, c, (action+1)%4)
    
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
                if (r == 0 and c == 3) or (r == 3 and c == 2):
                    continue
                next_utility[r][c] = max([calculate_utility(rewards, r, c, action) for action in range(NUM_ACTIONS)]) # Bellman update
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
            if (r == 0 and c == 3) or (r == 3 and c == 0):
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculate_utility(rewards, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy


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