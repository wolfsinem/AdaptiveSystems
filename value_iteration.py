import numpy as np

REWARD = -0.1
GAMMA = 0.7

ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # actions which can be D, L, U, R
NUM_ACTIONS = len(ACTIONS)

# Grid size based on amount of rows and columns
ROW = 4
COL = 4

# Grid with all of the rewards
rewards = [[0, 0, 0, 40],
           [0, 0, -10, -10],
           [0, 0, 0, 0],
           [10, -2, 0, 0]]


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
                val = "F +40"
            elif r == 3 and c == 0:
                val = "F +10"
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


def getU(rewards, r, c, action):
    """This function gets the utility of the state reached 
    by performing the given action from the given state
    """
    dr, dc = ACTIONS[action]
    newR = r + dr
    newC = c + dc
    if newR < 0 or newC < 0 or newR >= ROW or newC >= COL or (newR == newC == 1):
        return rewards[r][c]
    else:
        return rewards[newR][newC]


def calculateU(rewards, r, c, action):
    """This function calculates the utility of a state given 
    an action
    """

    # if r == 0 and c == 0:
    #     u = +10
    # elif r == 0 and c == 1:
    #     u = -2
    # elif r == 2 and c == 2:
    #     u = -10
    # elif r == 2 and c == 3:
    #     u = -10
    # elif r == 3 and c == 3:
    #     u = +40
    # else:
    u = REWARD
                
    u += 0.1 * GAMMA * getU(rewards, r, c, (action-1)%4)
    u += 0.7 * GAMMA * getU(rewards, r, c, action)
    u += 0.1 * GAMMA * getU(rewards, r, c, (action+1)%4)
    return u

def valueIteration(rewards):
    """This function initializes the maze grid with all of the 
    given rewards per state and prints it out nicely.    
    The grid has 4 rows and 4 columns which is set in the cell above.
    """
    print("During the value iteration:\n")
    while True:
        nextU = [[0, 0, 0, 40], 
                 [0, 0, 0, 0], 
                 [0, 0, 0, 0], 
                 [10, 0, 0, 0]]
        error = 0
        for r in range(ROW):
            for c in range(COL):
                if (r <= 1 and c == 4) or (r == 3 and c == 2):
                    continue
                nextU[r][c] = max([calculateU(rewards, r, c, action) for action in range(NUM_ACTIONS)]) # Bellman update
                error = max(error, abs(nextU[r][c]-rewards[r][c]))
        rewards = nextU
        maze_grid(rewards)
        if error < ((1-GAMMA) / GAMMA):
            break
    return rewards


def getOptimalPolicy(rewards):
    """This function gets the optimal policy from U
    """
    policy = [[-1, -1, -1, 40],[-1, -1, -10, -10],[-1, -1, -1, -1],[10, -2, -1, -1]]
    for r in range(ROW):
        for c in range(COL):
            if (r <= 1 and c == 4) or (r == 3 and c == 2):
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(rewards, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy


# Print the initial environment
print("The initial maze grid is:\n")
maze_grid(rewards)

# Value iteration
rewards = valueIteration(rewards)
# print(rewards)

# Get the optimal policy from U and print it
policy = getOptimalPolicy(rewards)
print("The optimal policy is:\n")
maze_grid(policy, True)