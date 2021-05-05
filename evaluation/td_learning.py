import numpy as np
import random

from maze_class import *
from coords import *

# Initialize parameters
MAX_EP = 5000
ALPHA = 0.01  # learning rate
ACTIONS = [0, 1, 2, 3]

# Initialize maze_rewards with 16 cells (one value per cell)
maze_rewards = np.zeros(16)
maze_rewards[3] = 40
maze_rewards[12] = 10
maze_rewards[13] = -2
maze_rewards[6] = -10
maze_rewards[7] = -10

# Initialize new maze_rewards which has 4 probabilities per cell (16x4)
q_maze_rewards = np.zeros((16, 4))
q_maze_rewards[3] = 40
q_maze_rewards[12] = 10
q_maze_rewards[13] = -2
q_maze_rewards[6] = -10
q_maze_rewards[7] = -10

def td_learning(env, discount_factor):
    """Temporal difference (TD) learning refers to a class of model-free
    reinforcement learning. This function implements that.

    Args:
        env ([type]): [description]
        discount_factor ([type]): [description]

    Returns:
        [type]: [description]
    """
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


def get_action(q_values, epsilon):
    """This function calculates the e-greedy which is used for the SARSA
    algorithm

    Args:
        q_values ([type]): [description]
        epsilon ([type]): [description]

    Returns:
        [type]: [description]
    """

    if random.random() > epsilon:
        return np.argmax(q_values)
    else:
        return random.randint(0, 3)


def sarsa(env, discount_factor, alpha):
    """Sarsa (on-policy TD control) for estimating Q ~ q*

    Args:
        env ([type]): [description]
        discount_factor ([type]): [description]
        alpha ([type]): [description]

    Returns:
        [type]: [description]
    """
    for ep in range(MAX_EP):
        iterator = 1
        state = env.reset()
        action = get_action(q_maze_rewards[state], 1)
        while True:
            next_state, reward, done = env.step(action)
            # Updating
            iterator += 1
            epsilon = 1 * 0.99 ** iterator
            next_action = get_action(q_maze_rewards[next_state], epsilon)
            q_maze_rewards[state][action] += alpha*(reward + discount_factor *
                                                    q_maze_rewards[next_state][next_action] - q_maze_rewards[state][action])
            state = next_state
            action = next_action
            if done:
                break
    return q_maze_rewards


def sarsaMAX(env, discount_factor, alpha):
    """Q-learning (SARSAMAX) is a model-free reinforcement learning algorithm to
    learn the value of an action in a particular state. This function implements
    that.

    Args:
        env ([type]): [description]
        discount_factor ([type]): [description]
        alpha ([type]): [description]

    Returns:
        [type]: [description]
    """
    for ep in range(MAX_EP):
        iterator = 1
        state = env.reset()
        while True:
            action = get_action(q_maze_rewards[state], 1 * 0.95**iterator)
            next_state, reward, done = env.step(action)

            # Updating
            q_maze_rewards[state][action] += alpha * (reward + discount_factor * max(
                q_maze_rewards[next_state]) - q_maze_rewards[state][action])

            iterator += 1
            state = next_state

            if done:
                break
    return q_maze_rewards
