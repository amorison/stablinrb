from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Callable, Optional, Sequence

    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def savefig(fig: Figure, name: str) -> None:
    fig.savefig(name, format="PDF", bbox_inches="tight")
    plt.close(fig)


def normalize(arr: NDArray) -> tuple[NDArray, np.complexfloating]:
    """Normalize complex array with element of higher modulus."""
    amax = arr[np.argmax(np.abs(arr))]
    return arr / amax, amax


def build_slices(
    i0n: Sequence[tuple[int, int]], imax: int
) -> tuple[
    Sequence[tuple[int, int]],
    Sequence[Callable],
    Sequence[slice],
    Sequence[slice],
    Sequence[slice],
    Sequence[slice],
]:
    """Build slices from a list of min/max index

    i0n: list of (imin, imax) indices tuples, one tuple per variable,
         without the boundaries if they are not needed
    imax: maximum index taking boundaries into account typically the
          number of Chebyshev points)
    """
    igf = [lambda idx, _=0: idx - i0n[0][0]]
    i_0s, i_ns = zip(*i0n)
    for i_0, i_n in zip(i_0s[1:], i_ns):
        ipn = igf[-1](i_n)
        igf.append(lambda idx, i_l=ipn - i_0 + 1: idx + i_l)
    slall = []  # entire vector
    slint = []  # interior points
    slgall = []  # entire vector with big matrix indexing
    slgint = []  # interior points with big matrix indexing
    for i_0, i_n, i_g in zip(i_0s, i_ns, igf):
        slall.append(slice(i_0, i_n + 1))
        slint.append(slice(1, imax))
        slgall.append(slice(i_g(i_0), i_g(i_n + 1)))
        slgint.append(slice(i_g(1), i_g(imax)))
    return i0n, igf, slall, slint, slgall, slgint
