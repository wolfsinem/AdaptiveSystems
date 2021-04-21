# Assignment for Adaptive Systems
import numpy as np

REWARD = -1
GAMMA = 0.9

# Arrow symbols
# ACTIONS = {
#     3: '\u2191', #U
#     2: '\u2192', #R
#     1: '\u2193', #D
#     0: '\u2190' #L
# }

# Initialize the actions which can be D, L, U, R
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] 
NUM_ACTIONS = len(ACTIONS)

# Grid size based on amount of rows and columns
ROW = 4
COL = 4

# Grid with all of the rewards
U = [[-1, -1, -1, 40],
    [-1, -1, -10, -10],
    [-1, -1, -1, -1],
    [10, -2, -1, -1]]

# Visualization
def maze_grid(arr, policy=False):
    """This function initializes the maze grid with all of the 
    given rewards per state and prints it out nicely.    
    The grid has 4 rows and 4 columns which is set in the cell above.

    Args:
        arr::[int]
            Multidimensional grid with rewards per state
    
    Returns:
        res::int
            Prints out the result
    """
    res = ""
    for r in range(ROW):
        res += "|"
        for c in range(COL):
            # val = "-1"
            # if r == 0 and c == 0:
            #     val = "+10" 
            # elif r == 0 and c == 1:
            #     val = "-2"
            # elif r == 2 and c == 2:
            #     val = "-10"
            # elif r == 2 and c == 3:
            #     val = "-10"
            # elif r == 3 and c == 3:
            #     val = "+40"
            # else:
            if policy:
                val = ['\u2193', '\u2190', '\u2191', '\u2192'][arr[r][c]]
            else:
                val = str(arr[r][c])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)


# Get the utility of the state reached by performing the given action from the given state
def getU(U, r, c, action):
    """This function initializes the maze grid with all of the 
    given rewards per state and prints it out nicely.    
    The grid has 4 rows and 4 columns which is set in the cell above.

    Args:
        U::[int]
            Multidimensional grid with rewards per state
        r::[int]
            Multidimensional grid with rewards per state          
        c::[int]
            Multidimensional grid with rewards per state
        action::[int]
            Multidimensional grid with rewards per state 
    Returns:
        U[newR][newC]::int
            Prints out the result
    """
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= ROW or newC >= COL or (newR == newC == 1):
        return U[r][c]
    else:
        return U[newR][newC]


# Calculate the utility of a state given an action
def calculateU(U, r, c, action):
    """This function initializes the maze grid with all of the 
    given rewards per state and prints it out nicely.    
    The grid has 4 rows and 4 columns which is set in the cell above.

    Args:
        U::[int]
            Multidimensional grid with rewards per state
        r::[int]
            Multidimensional grid with rewards per state          
        c::[int]
            Multidimensional grid with rewards per state
        action::[int]
            Multidimensional grid with rewards per state 
    Returns:
        u::int
            Prints out the result
    """
    u = REWARD
    u += 0.1 * GAMMA * getU(U, r, c, (action-1)%4)
    u += 0.8 * GAMMA * getU(U, r, c, action)
    u += 0.1 * GAMMA * getU(U, r, c, (action+1)%4)
    return u


def valueIteration(U):
    """This function initializes the maze grid with all of the 
    given rewards per state and prints it out nicely.    
    The grid has 4 rows and 4 columns which is set in the cell above.

    Args:
        U::[int]
            Multidimensional grid with rewards per state
    
    Returns:
        U::int
            Prints out the result
    """
    print("During the value iteration:\n")
    while True:
        nextU = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
        error = 0
        for r in range(ROW):
            for c in range(COL):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                nextU[r][c] = max([calculateU(U, r, c, action) for action in range(NUM_ACTIONS)]) # Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        maze_grid(U)
        if error < ((1-GAMMA) / GAMMA):
            break
    return U


# Get the optimal policy from U
def getOptimalPolicy(U):
    """This function initializes the maze grid with all of the 
    given rewards per state and prints it out nicely.    
    The grid has 4 rows and 4 columns which is set in the cell above.

    Args:
        U::[int]
            Multidimensional grid with rewards per state
    
    Returns:
        policy::int
            returns the optimal policy
    """
    policy = [[-1, -1, -1, -1] for i in range(ROW)]
    for r in range(ROW):
        for c in range(COL):
            if (r <= 1 and c == 3) or (r == c == 1):
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(U, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy