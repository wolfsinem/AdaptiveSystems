import numpy as np

maze_coords = {
    "[0, 0]": 0, "[0, 1]": 1, "[0, 2]": 2, "[0, 3]": 3,
    "[1, 0]": 4, "[1, 1]": 5, "[1, 2]": 6, "[1, 3]": 7,
    "[2, 0]": 8, "[2, 1]": 9, "[2, 2]": 10, "[2, 3]": 11,
    "[3, 0]": 12, "[3, 1]": 13, "[3, 2]": 14, "[3, 3]": 15
}

reversed_maze = {
    "0": [0, 0], "1": [0, 1], "2": [0, 2], "3": [0, 3],
    "4": [1, 0], "5": [1, 1], "6": [1, 2], "7": [1, 3],
    "8": [2, 0], "9": [2, 1], "10": [2, 2], "11": [2, 3],
    "12": [3, 0], "13": [3, 1], "14": [3, 2], "15": [3, 3],
}

# coordinations for the q table visualization
top_coords = [(0.38, 0.25), (1.38, 0.25), (2.38, 0.25), (3.38, 0.25),
              (0.38, 1.25), (1.38, 1.25), (2.38, 1.25), (3.38, 1.25),
              (0.38, 2.25), (1.38, 2.25), (2.38, 2.25), (3.38, 2.25),
              (0.38, 3.25), (1.38, 3.25), (2.38, 3.25), (3.38, 3.25)]

right_coords = [(0.65, 0.5), (1.65, 0.5), (2.65, 0.5), (3.65, 0.5),
                (0.65, 1.5), (1.65, 1.5), (2.65, 1.5), (3.65, 1.5),
                (0.65, 2.5), (1.65, 2.5), (2.65, 2.5), (3.65, 2.5),
                (0.65, 3.5), (1.65, 3.5), (2.65, 3.5), (3.65, 3.5)]

bottom_coords = [(0.38, 0.8), (1.38, 0.8), (2.38, 0.8), (3.38, 0.8),
                 (0.38, 1.8), (1.38, 1.8), (2.38, 1.8), (3.38, 1.8),
                 (0.38, 2.8), (1.38, 2.8), (2.38, 2.8), (3.38, 2.8),
                 (0.38, 3.8), (1.38, 3.8), (2.38, 3.8), (3.38, 3.8)]

left_coords = [(0.05, 0.5), (1.05, 0.5), (2.05, 0.5), (3.05, 0.5),
               (0.05, 1.5), (1.05, 1.5), (2.05, 1.5), (3.05, 1.5),
               (0.05, 2.5), (1.05, 2.5), (2.05, 2.5), (3.05, 2.5),
               (0.05, 3.5), (1.05, 3.5), (2.05, 3.5), (3.05, 3.5)]
