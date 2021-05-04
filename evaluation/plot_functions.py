import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_grid(evaluation):
    """This function visualizes the values for each state/grid

    Args:
        evaluation ([type]): [description]
    """
    plt.figure(figsize=(10, 5))
    p = sns.heatmap(evaluation, cmap='plasma', annot=True,
                    fmt=".1f", annot_kws={'size': 16}, square=True)
    p.set_ylim(len(evaluation)+0.01, -0.01)


def q_table(left, bottom, right, top, ax=None, plots_={}, colors_={}):
    """This function is initializing the grid for the q_table taking the
    left, bottom, right and top values

    Args:
        left ([type]): left side of the grid
        bottom ([type]): bottom of the grid
        right ([type]): right side of the grid
        top ([type]): top of the grid
        ax ([type], optional): [description]. Defaults to None.
        plot_ (dict, optional): [description]. Defaults to {}.
        color_ (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """

    if not ax:
        ax = plt.gca()

    # Get the amount of rows and columns of the grid
    n_rows = left.shape[0]
    n_cols = left.shape[1]

    # Create 5x2 and 4x3 arrays
    a = np.array([[0, 0], [0, 1], [.5, .5], [1, 0], [1, 1]])
    extended_cell = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])

    A = np.zeros((n_rows * n_cols * 5, 2))
    Tr = np.zeros((n_rows * n_cols * 4, 3))

    for i in range(n_rows):
        for j in range(n_cols):
            n_cells = i * n_cols + j
            A[n_cells * 5:(n_cells + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
            Tr[n_cells * 4:(n_cells + 1) * 4, :] = extended_cell + n_cells * 5

    # Add every part of the grid together with rewards
    expanded_reward_grid = np.c_[left.flatten(), bottom.flatten(),
              right.flatten(), top.flatten()].flatten()

    triplot = ax.triplot(A[:, 0], A[:, 1], Tr, **plots_)
    tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr,
                             facecolors=expanded_reward_grid, **colors_)
    return tripcolor


def plot_q_table(maze_rewards):
    """This function plots the q table, which is initialized in the previous
    function

    Args:
        maze_rewards ([type]): [description]
    """

    top = maze_rewards[:, 0].reshape((4, 4))
    top_coords = [(0.38, 0.25), (1.38, 0.25), (2.38, 0.25), (3.38, 0.25),
                           (0.38, 1.25), (1.38, 1.25), (2.38, 1.25), (3.38, 1.25),
                           (0.38, 2.25), (1.38, 2.25), (2.38, 2.25), (3.38, 2.25),
                           (0.38, 3.25), (1.38, 3.25), (2.38, 3.25), (3.38, 3.25)]

    right = maze_rewards[:, 1].reshape((4, 4))
    right_coords = [(0.65, 0.5), (1.65, 0.5), (2.65, 0.5), (3.65, 0.5),
                             (0.65, 1.5), (1.65, 1.5), (2.65, 1.5), (3.65, 1.5),
                             (0.65, 2.5), (1.65, 2.5), (2.65, 2.5), (3.65, 2.5),
                             (0.65, 3.5), (1.65, 3.5), (2.65, 3.5), (3.65, 3.5)]

    bottom = maze_rewards[:, 2].reshape((4, 4))
    bottom_coords = [(0.38, 0.8), (1.38, 0.8), (2.38, 0.8), (3.38, 0.8),
                              (0.38, 1.8), (1.38, 1.8), (2.38, 1.8), (3.38, 1.8),
                              (0.38, 2.8), (1.38, 2.8), (2.38, 2.8), (3.38, 2.8),
                              (0.38, 3.8), (1.38, 3.8), (2.38, 3.8), (3.38, 3.8)]

    left = maze_rewards[:, 3].reshape((4, 4))
    left_coords = [(0.05, 0.5), (1.05, 0.5), (2.05, 0.5), (3.05, 0.5),
                            (0.05, 1.5), (1.05, 1.5), (2.05, 1.5), (3.05, 1.5),
                            (0.05, 2.5), (1.05, 2.5), (2.05, 2.5), (3.05, 2.5),
                            (0.05, 3.5), (1.05, 3.5), (2.05, 3.5), (3.05, 3.5)]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.set_ylim(4, 0)
    tripcolor = q_table(left, top, right, bottom, ax=ax,
                        plots_={"color": "k", "lw": 1},
                        colors_={"cmap": "coolwarm"})

    ax.margins(0)
    ax.set_aspect("equal")
    fig.colorbar(tripcolor)

    for i, (xi, yi) in enumerate(top_coords):
        plt.text(xi, yi, round(top.flatten()[i], 2), size=11, color="w")

    for i, (xi, yi) in enumerate(right_coords):
        plt.text(xi, yi, round(right.flatten()[i], 2), size=11, color="w")

    for i, (xi, yi) in enumerate(left_coords):
        plt.text(xi, yi, round(left.flatten()[i], 2), size=11, color="w")

    for i, (xi, yi) in enumerate(bottom_coords):
        plt.text(xi, yi, round(bottom.flatten()[i], 2), size=11, color="w")

    plt.show()
