from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

CLUE_FACE = "#e6e6e6"
FILL_COLOR = "#1f4e9c"
WRONG_COLOR = "#d62728"


def _as_boards(x, name):
    if x is None:
        return None
    if isinstance(x, np.ndarray) and x.shape in ((9, 9), (81,)):
        x = [x]
    boards = [np.asarray(b).reshape(9, 9) for b in x]
    for b in boards:
        if not np.isin(b, np.arange(10)).all():
            raise ValueError(f"{name}: Sudoku values must be integers 0-9")
    return boards


def show(x, outfile=None, titles=None, show_plot=True, clues=None, target=None,
         ncols=None, legend=None, suptitle=None, cell_size=0.55):
    """Visualize one or multiple Sudoku boards (9x9 or flat 81, 0 = blank).

    clues:  boards whose non-zero cells are the given clues (shaded, bold).
            Other non-zero cells in x are drawn as filled-in digits.
    target: solution boards; filled-in cells differing from it are marked red.
    """
    boards = _as_boards(x, "x")
    if not boards:
        raise ValueError("No Sudoku boards provided")
    clues = _as_boards(clues, "clues")
    target = _as_boards(target, "target")
    for name, extra in (("clues", clues), ("target", target)):
        if extra is not None and len(extra) != len(boards):
            raise ValueError(f"Number of {name} boards must match x")

    if titles is None:
        titles = [f"Sudoku {i + 1}" for i in range(len(boards))]
    if len(titles) != len(boards):
        raise ValueError("Number of titles must match boards")

    ncols = ncols or min(len(boards), 4)
    nrows = -(-len(boards) // ncols)
    side = 9 * cell_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(side * ncols, (side + 0.6) * nrows), squeeze=False)
    for ax in axes.ravel()[len(boards):]:
        ax.axis("off")

    for i, (board, ax, title) in enumerate(zip(boards, axes.ravel(), titles)):
        given = clues[i] > 0 if clues is not None else board > 0
        wrong = (board != target[i]) & ~given & (board > 0) if target is not None else np.zeros_like(given)

        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=9, pad=6)

        for row in range(9):
            for col in range(9):
                if given[row, col]:
                    ax.add_patch(Rectangle((col, row), 1, 1, facecolor=CLUE_FACE, linewidth=0))
                elif wrong[row, col]:
                    ax.add_patch(Rectangle((col, row), 1, 1, facecolor=WRONG_COLOR, alpha=0.25, linewidth=0))

                value = int(board[row, col])
                if value != 0:
                    ax.text(
                        col + 0.5, row + 0.52, str(value), ha="center", va="center",
                        fontsize=11 * cell_size / 0.55,
                        fontweight="bold" if given[row, col] else "normal",
                        color="black" if given[row, col] else (WRONG_COLOR if wrong[row, col] else FILL_COLOR),
                    )

        # Draw grid lines
        for i in range(10):
            width = 2.5 if i % 3 == 0 else 0.6

            ax.plot([i, i], [0, 9],
                    color="black", linewidth=width)
            ax.plot([0, 9], [i, i],
                    color="black", linewidth=width)

    if legend is None:
        legend = clues is not None or target is not None
    if legend:
        handles = [Patch(facecolor=CLUE_FACE, edgecolor="black", label="given clue"),
                   Patch(facecolor="white", edgecolor=FILL_COLOR, label="filled in")]
        if target is not None:
            handles.append(Patch(facecolor=WRONG_COLOR, alpha=0.25, edgecolor=WRONG_COLOR, label="incorrect"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.04)

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