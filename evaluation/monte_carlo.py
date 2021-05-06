import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

from maze_class import *
from coords import *

# Initialise parameters
MAX_EP = 100
ACTIONS = [0, 1, 2, 3]
STEP_COST = -1
MAX_EP_LEN = 30

EPSILON = 0.25

# intialize maze rewards grid with values
mc_maze_rewards = np.zeros(16)
mc_maze_rewards[3] = 40
mc_maze_rewards[12] = 10
mc_maze_rewards[13] = -2
mc_maze_rewards[6] = -10
mc_maze_rewards[7] = -10

fv_maze_rewards = np.zeros((16, 4))
fv_maze_rewards[3] = 40
fv_maze_rewards[12] = 10
fv_maze_rewards[13] = -2
fv_maze_rewards[6] = -10
fv_maze_rewards[7] = -10

def mc_evaluation_policy(env, discount_factor):
    """First visit Monte Carlo prediction for estimating V ~ v pi.

    Args:
        env ([type]): Init environment
        discount_factor ([type]): The discount factor

    Returns:
        [type]: An array with rewards
    """
    rewards = {
        "0": [], "1": [], "2": [], "3": [],
        "4": [], "5": [], "6": [], "7": [],
        "8": [], "9": [], "10": [], "11": [],
        "12": [], "13": [], "14": [], "15": []
    }
    for ep in range(MAX_EP):
        G = 0
        start_state = env.reset()
        episode = []
        while True:
            action = random.choice(ACTIONS)
            next_state, reward, done = env.step(action)
            episode.append((start_state, reward))
            start_state = next_state
            if done:
                break

        for i, step in enumerate(episode[::-1]):
            G = discount_factor * G + step[1]
            if step[0] not in np.array(episode[::-1])[:, 0][i+1:]:
                rewards[str(step[0])].append(G)
                mc_maze_rewards[step[0]] = round(
                    np.mean(rewards[str(step[0])]), 2)

    return mc_maze_rewards


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


def rewards_lists(cells, actions):
    """Initialize rewards lists

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

    rewards = {}
    for state in possible_states:
        for action in possible_actions:
            rewards[state+", "+action] = []

    return rewards

def first_visit_mc(env, max_ep, discount_factor, rewards):
    """On-policy first-visit MC control (for e-soft policies), estimates
    pi ~ pi*

    Args:
        env ([type]): Initialize environment
        max_ep ([type]): Maximum episodes
        discount_factor ([type]): The discount factor
        rewards ([type]): Given rewards

    Returns:
        [type]: Array with rewards
    """
    for ep in range(max_ep):
        G = 0
        state = env.reset()
        trajectory = []
        while True:
            action_values = fv_maze_rewards[state]
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
                rewards[str(step[0])+", "+str(step[1])].append(G)
                fv_maze_rewards[step[0]][step[1]] = np.mean(
                    rewards[str(step[0])+", "+str(step[1])])
    return fv_maze_rewards


def print_val(values, length=4):
    """Basic function to print out the values in a 'grid'.

    Args:
        values ([type]): The rewards
        length (int, optional): Lenght of grid. Defaults to 4.
    """
    for i in range(len(values) // length):
        sub_list = values[i * length:(i + 1) * length]
        print(' | '.join(map(str, sub_list)))
