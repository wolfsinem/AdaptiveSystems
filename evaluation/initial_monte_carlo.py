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


def print_val(values, length=4):
    """Basic function to print out the values in a 'grid'.

    Args:
        values ([type]): [description]
        length (int, optional): [description]. Defaults to 4.
    """
    for i in range(len(values) // length):
        sub_list = values[i * length:(i + 1) * length]
        print(' | '.join(map(str, sub_list)))


def plot_val(state_value_grid):
    """[summary]

    Args:
        state_value_grid ([type]): [description]
    """
    plt.figure(figsize=(10, 5))
    p = sns.heatmap(state_value_grid, cmap='coolwarm', annot=True,
                    fmt=".1f", annot_kws={'size': 16}, square=True)
    p.set_ylim(len(state_value_grid)+0.01, -0.01)


env = Maze(maze_coords,
           reversed_maze, step_cost=STEP_COST, max_ep_length=MAX_EP_LEN)


# print(f"Evaluation with discount factor 0.99:")
# print(print_val(mc_evaluation_policy(env, discount_factor=0.99)))
# plot_val(mc_evaluation_policy(env, discount_factor=0.99))

# print(f"Evaluation with discount factor 1:")
# print(print_val(mc_evaluation_policy(env, discount_factor=1)))
# plot_val(mc_evaluation_policy(env, discount_factor=1))
