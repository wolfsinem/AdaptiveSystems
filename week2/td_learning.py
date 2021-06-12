import numpy as np
import random

from maze_class import Maze
from coords import maze_coords, reversed_maze

# Initialize parameters
MAX_EP = 5000
ALPHA = 0.01  # learning rate
ACTIONS = [0, 1, 2, 3] # all possible actions such as up and down...

# Initialize state value grid with 16 cells (one value per cell)
# For TD learning
state_val_grid = np.zeros(16)
state_val_grid[3] = 40
state_val_grid[12] = 10
state_val_grid[13] = -2
state_val_grid[6] = -10
state_val_grid[7] = -10

# Initialize new state value grid which has 4 probabilities per cell (16x4)
# For Q-Learning
state_val_grid_ = np.zeros((16, 4))
state_val_grid_[3] = 40
state_val_grid_[12] = 10
state_val_grid_[13] = -2
state_val_grid_[6] = -10
state_val_grid_[7] = -10

def td_learning(env, discount_factor):
    """Temporal difference (TD) learning refers to a class of model-free
    reinforcement learning. This function implements that.

    Args:
        env ([type]): [the environment]
        discount_factor ([type]): [discount factor]

    Returns:
        [type]: [description]
    """
    for ep in range(MAX_EP):
        G = 0
        state = env.reset()
        while True:
            action = random.choice(ACTIONS)  # Choose random policy
            next_state, reward, done = env.step(action)
            # updating the state value grid
            state_val_grid[state] += ALPHA * \
                (reward + discount_factor *
                 state_val_grid[next_state] - state_val_grid[state])
            state = next_state
            if done:
                break

    return state_val_grid


def get_action(q_values, epsilon):
    """This function calculates the e-greedy which is used for the SARSA
    algorithm.

    Args:
        q_values ([type]): [the Q values]
        epsilon ([type]): [epsilon]

    Returns:
        [int]: [e-greedy]
    """

    if random.random() > epsilon:
        return np.argmax(q_values)
    else:
        return random.randint(0, 3)


def sarsa(env, discount_factor, alpha):
    """Sarsa (on-policy TD control) for estimating Q ~ q*

    Args:
        env ([type]): [the environment]
        discount_factor ([type]): [discount factor]
        alpha ([type]): [alpha]

    Returns:
        [type]: [state value grid]
    """
    for ep in range(MAX_EP):
        iterator = 1
        state = env.reset()
        action = get_action(state_val_grid_[state], 1)
        while True:
            next_state, reward, done = env.step(action)
            # Updating
            iterator += 1
            epsilon = 1 * 0.99 ** iterator
            next_action = get_action(state_val_grid_[next_state], epsilon)
            state_val_grid_[state][action] += alpha*(reward + discount_factor *
                                                    state_val_grid_[next_state][next_action] - state_val_grid_[state][action])
            state = next_state
            action = next_action
            if done:
                break
    return state_val_grid_


def sarsaMAX(env, discount_factor, alpha):
    """Q-learning (SARSAMAX) is a model-free reinforcement learning algorithm to
    learn the value of an action in a particular state. This function implements
    that.

    Args:
        env ([type]): [the environment]
        discount_factor ([type]): [discount factor]
        alpha ([type]): [alpha]

    Returns:
        [type]: [state value grid]
    """
    for ep in range(MAX_EP):
        iterator = 1
        state = env.reset()
        while True:
            action = get_action(state_val_grid_[state], 1 * 0.95**iterator)
            next_state, reward, done = env.step(action)
            # Updating
            state_val_grid_[state][action] += alpha * (reward + discount_factor * max(
                state_val_grid_[next_state]) - state_val_grid_[state][action])
            iterator += 1
            state = next_state
            if done:
                break
    return state_val_grid_
