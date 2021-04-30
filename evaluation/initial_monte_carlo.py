import copy
import numpy as np
import random

from maze_class import *
from coords import *

# Initialise parameters
MAX_EP = 100
DISCOUNT_FACTOR = 0.99
ACTIONS = [0, 1, 2, 3]
STEP_COST = -1
MAX_EP_LEN = 30


def mc_evaluation_policy(env):
    """[summary]

    Args:
        env ([type]): [description]

    Returns:
        [type]: [description]
    """
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
            G = DISCOUNT_FACTOR * G + step[1]
            if step[0] not in np.array(episode[::-1])[:, 0][i+1:]:
                rewards[str(step[0])].append(G)
                maze_rewards[step[0]] = np.mean(rewards[str(step[0])])

    return maze_rewards


env = Maze(maze_coords,
           reversed_maze, step_cost=STEP_COST, max_ep_length=MAX_EP_LEN)


print(mc_evaluation_policy(env=env))
