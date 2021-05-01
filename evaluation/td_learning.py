import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

from maze_class import *
from coords import *

# Initialise parameters
MAX_EP = 100
ALPHA = 0.01  # learning rate
ACTIONS = [0, 1, 2, 3]

def td_learning(env, discount_factor):
    """Temporal difference (TD) learning refers to a class of model-free
    reinforcement learning. This function implements that.

    Args:
        env ([type]): [description]
        discount_factor ([type]): [description]

    Returns:
        [type]: [description]
    """

    maze_rewards = np.zeros(16)
    maze_rewards[3] = 40
    maze_rewards[12] = 10
    maze_rewards[13] = -2
    maze_rewards[6] = -10
    maze_rewards[7] = -10

    for ep in range(MAX_EP):
        G = 0
        state = env.reset()
        while True:
            action = random.choice(ACTIONS)  # random policy
            next_state, reward, done = env.step(action)

            # Updating
            maze_rewards[state] += ALPHA * \
                (reward + discount_factor *
                 maze_rewards[next_state] - maze_rewards[state])
            state = next_state
            if done:
                break

    return maze_rewards
