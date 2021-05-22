import numpy as np
import copy

class Maze():
    """Class for a 4x4 maze with 16 cells in total from 0 to 15.
        [0 , 1 , 2 , 3]
        [4 , 5 , 6 , 7]
        [8 , 9 , 10, 11]
        [12 , 13 , 14 , 15]
        """
    def __init__(self, maze_coords, reversed_maze,
                 step_cost=-1, max_ep_length=100, es=False):
        """[summary]

        Args:
            maze_coords ([type]): [description]
            reversed_maze ([type]): [description]
            step_cost (int, optional): [description]. Defaults to -1.
            max_ep_length (int, optional): [description]. Defaults to 100.
            es (bool, optional): [description]. Defaults to False.
        """

        self.actions = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3,
        }

        self.list_actions = list(self.actions.values())
        self.n_actions = len(self.list_actions)

        self.row = 4
        self.col = 4
        self.grid = np.zeros((self.row,self.col))

        # TERMINAL STATES
        self.grid[0][3] = 0
        self.grid[3][0] = 0

        # WATER STATES
        self.grid[1][2] = -10
        self.grid[1][3] = -10

        # ENEMY STATE
        self.grid[3][1] = -2

        self.exploring_starts = es

        self.terminal_state1 = 3
        self.terminal_state2 = 12
        self.start_state = 14
        self.water = [6,7]
        self.enemy = 13

        self.done = False
        self.max_ep_length = max_ep_length
        self.steps = 0
        self.step_cost = step_cost
        self.maze_coords = maze_coords
        self.reversed_maze = reversed_maze


    def reset(self):
        """Resets the state to it's initial values.

        Returns:
            [type]: Returns new exploring start
        """
        self.done = False
        self.steps = 0
        self.grid = np.zeros((4,4))

        self.grid[0][3] = 0
        self.grid[3][0] = 0

        # WATER STATES
        self.grid[1][2] = -10
        self.grid[1][3] = -10

        # ENEMY STATE
        self.grid[3][1] = -2

        if self.exploring_starts:
            self.start_state = np.random.choice(
                [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16])
        else:
            self.start_state = 14
        return self.start_state


    def get_next_state(self, current_position, action):
        """Given the current position, retrieve the next state based on an
        action.

        Args:
            current_position ([type]): current position on the grid
            action ([type]): Type of action the agent takes e.g. U/D/L/R

        Returns:
            [type]: The coordinates of the next state
        """

        next_state = self.reversed_maze[str(current_position)].copy()

        # If an action is equal to 0 (up), and the first index on row 0 is not
        # equal to 0 (because there is a wall), extract with 1 from the first
        # index
        if action == 0 and next_state[0] != 0:
            next_state[0] -= 1

        elif action == 1 and next_state[1] != 3:
            next_state[1] += 1

        elif action == 2 and next_state[0] != 3:
            next_state[0] += 1  # down

        elif action == 3 and next_state[1] != 0:
            next_state[1] -= 1  # left

        else:
            pass

        return self.maze_coords[str(next_state)]


    def step(self, action):
        """Takes a step and based on that give back a reward.

        Args:
            action ([type]): Type of action the agent takes e.g. U/D/L/R

        Returns:
            [type]: The next state and given reward
        """
        current_position = self.start_state
        next_state = self.get_next_state(current_position, action)

        self.steps += 1

        if next_state == self.terminal_state1:
            reward = 40
            self.done = True

        elif next_state == self.terminal_state2:
            reward = 10
            self.done = True

        elif next_state == any(self.water):
            reward = -10

        elif next_state == self.enemy:
            reward = -2

        else:
            reward = self.step_cost

        if self.steps == self.max_ep_length:
            self.done = True

        self.start_state = next_state
        return next_state, reward, self.done
