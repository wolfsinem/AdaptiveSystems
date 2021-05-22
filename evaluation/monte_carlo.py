from maze_class import Maze
from coords import maze_coords, reversed_maze
import sys
sys.path.append('/Users/wolfsinem/as/evaluation/')

import numpy as np
import random

# Initialise parameters
MAX_EP = 100
ACTIONS = [0, 1, 2, 3]
STEP_COST = -1
MAX_EP_LEN = 30

EPSILON = 0.25

# intialize state value grid
state_val_grid = np.zeros(16)
state_val_grid[3] = 0
state_val_grid[12] = 0
state_val_grid[13] = -2
state_val_grid[6] = -10
state_val_grid[7] = -10

state_val_grid_fv = np.zeros((16, 4))
state_val_grid_fv[3] = 0
state_val_grid_fv[12] = 0
state_val_grid_fv[13] = -2
state_val_grid_fv[6] = -10
state_val_grid_fv[7] = -10

# Initialize environment
env = Maze(maze_coords, reversed_maze,
           step_cost=STEP_COST, max_ep_length=MAX_EP_LEN)

def mc_evaluation_policy(env, discount_factor):
    """First visit Monte Carlo prediction for estimating V ~ v pi.

    Args:
        env ([type]): Init environment
        discount_factor ([type]): The discount factor

    Returns:
        [type]: An array with returns
    """
    returns = {
        "0": [], "1": [], "2": [], "3": [],
        "4": [], "5": [], "6": [], "7": [],
        "8": [], "9": [], "10": [], "11": [],
        "12": [], "13": [], "14": [], "15": []
    }
    for ep in range(MAX_EP):
        G = 0
        start_state = env.reset()
        # generate an episode
        # an episode is a list of (state,action,reward) tuples
        episode = []
        while True:
            action = random.choice(ACTIONS) # choose random policy
            next_state, reward, done = env.step(action)
            episode.append((start_state, reward))
            start_state = next_state
            if done:
                break
        for i, step in enumerate(episode[::-1]):
            G = discount_factor * G + step[1]
            # check the first visit
            if step[0] not in np.array(episode[::-1])[:, 0][i+1:]:
                returns[str(step[0])].append(G)
                # calculate the average return and add to returns dict
                state_val_grid[step[0]] = round(
                    np.mean(returns[str(step[0])]), 2)

    return state_val_grid


def probability(A):
    """This function calculates the probability of an agent going either
    UP, DOWN, RIGHT or UP.

    Args:
        A ([type]): [description]

    Returns:
        [type]: The probabilty
    """
    idx = np.argmax(A)
    probabilities = []
    action_val = np.sqrt(sum([i**2 for i in A]))

    if action_val == 0:
        action_val = 1.0
    for i, a in enumerate(A):
        if i == idx:
            probabilities.append(round(1-EPSILON + (EPSILON/action_val), 3))
        else:
            probabilities.append(round(EPSILON/action_val, 3))
    err = sum(probabilities)-1
    substracter = err/len(A)

    return np.array(probabilities)-substracter


def returns_lists(cells=16, actions=4):
    """Initialize returns list. Each state has four possible actions.

    Args:
        cells ([type]): Array with cells in grid
        actions ([type]): Array with possible actions

    Returns:
        [type]: Rewards dictionary
    """
    # Each state has four possible actions to take
    def create_array(n, lst):
        for i in range(n):
            lst.append(str(i))
        return lst

    possible_states = []
    possible_states = create_array(cells, possible_states)

    possible_actions = []
    possible_actions = create_array(actions, possible_actions)

    returns = {}
    for state in possible_states:
        for action in possible_actions:
            returns[state+", "+action] = []

    return returns

def first_visit_mc(env, max_ep, discount_factor, returns):
    """On-policy first-visit MC control (for e-soft policies), estimates
    pi ~ pi*

    Args:
        env ([type]): Initialize environment
        max_ep ([type]): Maximum episodes
        discount_factor ([type]): The discount factor
        returns ([type]): Given returns

    Returns:
        [type]: Array with rewards
    """
    for ep in range(max_ep):
        G = 0
        state = env.reset()
        # generate trajectory
        # each item is as follows (state,action,reward)
        trajectory = []
        while True:
            action_values = state_val_grid_fv[state]
            probs = probability(action_values)
            action = np.random.choice(np.arange(4), p=probs)
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            if done:
                break
        for idx, step in enumerate(trajectory[::-1]):
            G = discount_factor * G + step[2]
            # first visit check
            if step[0] not in np.array(trajectory[::-1])[:, 0][idx+1:]:
                returns[str(step[0])+", "+str(step[1])].append(G)
                # calculate the average return
                state_val_grid_fv[step[0]][step[1]] = np.mean(
                    returns[str(step[0])+", "+str(step[1])])
    return state_val_grid_fv


def print_val(values, length=4):
    """Basic function to print out the values in a 'grid'.

    Args:
        values ([type]): The rewards
        length (int, optional): Lenght of grid. Defaults to 4.
    """
    for i in range(len(values) // length):
        sub_list = values[i * length:(i + 1) * length]
        print(' | '.join(map(str, sub_list)))


# returns = returns_lists()
# print(first_visit_mc(env=env, max_ep=MAX_EP, discount_factor=1, returns=returns))
