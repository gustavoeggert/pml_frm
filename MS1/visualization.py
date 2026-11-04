from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def show(x, outfile=None, titles=None, show_plot=True):
    """Visualize one or multiple Sudoku boards."""

    # Accept a single board or a list of boards
    if isinstance(x, np.ndarray) and x.shape == (9, 9):
        boards = [x]
    else:
        boards = list(x)

    if not boards:
        raise ValueError("No Sudoku boards provided")

    if titles is None:
        titles = [f"Sudoku {i + 1}" for i in range(len(boards))]

    if len(titles) != len(boards):
        raise ValueError("Number of titles must match boards")

    fig, axes = plt.subplots(
        1,
        len(boards),
        figsize=(5 * len(boards), 5),
        squeeze=False,
    )

    for board, ax, title in zip(boards, axes[0], titles):
        board = np.asarray(board)

        if board.shape != (9, 9):
            raise ValueError("Each Sudoku board must be 9x9")

        if not np.isin(board, np.arange(10)).all():
            raise ValueError("Sudoku values must be integers 0-9")

        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=15, pad=15)

        # Draw Sudoku numbers
        for row in range(9):
            for col in range(9):
                value = int(board[row, col])

                if value != 0:
                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=16,
                    )

        # Draw grid lines
        for i in range(10):
            width = 2.5 if i % 3 == 0 else 0.6

            ax.plot([i, i], [0, 9],
                    color="black", linewidth=width)
            ax.plot([0, 9], [i, i],
                    color="black", linewidth=width)

    fig.tight_layout()

    # Save visualization if requested
    if outfile is not None:
        path = Path(outfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig