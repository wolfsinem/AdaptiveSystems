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

epsilon = 0.25

def mc_evaluation_policy(env, discount_factor):
    """[summary]

    Args:
        env ([type]): [description]

    Returns:
        [type]: [description]
    """
    rewards = {
        "0": [], "1": [], "2": [], "3": [],
        "4": [], "5": [], "6": [], "7": [],
        "8": [], "9": [], "10": [], "11": [],
        "12": [], "13": [], "14": [], "15": []
    }

    maze_rewards = np.zeros(16)
    maze_rewards[3] = 40
    maze_rewards[12] = 10
    maze_rewards[13] = -2
    maze_rewards[6] = -10
    maze_rewards[7] = -10

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
                maze_rewards[step[0]] = round(
                    np.mean(rewards[str(step[0])]), 2)

    return maze_rewards


def probability(A):
    """[summary]

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    idx = np.argmax(A)
    probabilities = []
    action_val = np.sqrt(sum([i**2 for i in A]))

    if action_val == 0:
        action_val = 1.0
    for i, a in enumerate(A):
        if i == idx:
            probabilities.append(round(1-epsilon + (epsilon/action_val), 3))
        else:
            probabilities.append(round(epsilon/action_val, 3))
    err = sum(probabilities)-1
    substracter = err/len(A)

    return np.array(probabilities)-substracter


def rewards_lists(cells, actions):
    """Initialize rewards lists

    Args:
        cells ([type]): [description]
        actions ([type]): [description]

    Returns:
        [type]: [description]
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
        env ([type]): [description]
        max_ep ([type]): [description]
        discount_factor ([type]): [description]
        rewards ([type]): [description]

    Returns:
        [type]: [description]
    """

    maze_rewards = np.zeros((16, 4))
    maze_rewards[3] = 40
    maze_rewards[12] = 10
    maze_rewards[13] = -2
    maze_rewards[6] = -10
    maze_rewards[7] = -10

    for ep in range(max_ep):
        G = 0
        state = env.reset()
        trajectory = []
        while True:
            action_values = maze_rewards[state]
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
                maze_rewards[step[0]][step[1]] = np.mean(
                    rewards[str(step[0])+", "+str(step[1])])
    return maze_rewards


def print_val(values, length=4):
    """Basic function to print out the values in a 'grid'.

    Args:
        values ([type]): [description]
        length (int, optional): [description]. Defaults to 4.
    """
    for i in range(len(values) // length):
        sub_list = values[i * length:(i + 1) * length]
        print(' | '.join(map(str, sub_list)))
