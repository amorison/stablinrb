from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import numpy as np

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def savefig(fig: Figure, name: str) -> None:
    fig.savefig(name, format="PDF", bbox_inches="tight")
    plt.close(fig)


def normalize(arr: NDArray) -> tuple[NDArray, np.complexfloating]:
    """Normalize complex array with element of higher modulus."""
    amax = arr[np.argmax(np.abs(arr))]
    return arr / amax, amax
