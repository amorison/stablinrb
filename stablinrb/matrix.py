from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import numpy as np

if typing.TYPE_CHECKING:
    from typing import Mapping, Sequence

    from numpy.typing import NDArray


@dataclass(frozen=True)
class Slices:
    include_bnd: Mapping[str, tuple[bool, bool]]  # top, bot
    nnodes: int

    @cached_property
    def variables(self) -> Sequence[str]:
        return tuple(self.include_bnd.keys())

    @cached_property
    def _lengths(self) -> Sequence[int]:
        """Length of each chunk, one chunk per variable."""
        return tuple(
            (self.nnodes - 2 + sum(self.include_bnd[var]) for var in self.variables)
        )

    @property
    def total_size(self) -> int:
        return sum(self._lengths)

    @cached_property
    def _var_to_len(self) -> Mapping[str, int]:
        return dict(zip(self.variables, self._lengths))

    @cached_property
    def _var_to_i0(self) -> Mapping[str, int]:
        cumlen = np.zeros(len(self.variables), dtype=np.int64)
        cumlen[1:] = np.cumsum(self._lengths[:-1])
        return dict(zip(self.variables, cumlen))

    def bulk(self, var: str) -> slice:
        imin = self._var_to_i0[var] + self.include_bnd[var][0]
        imax = self._var_to_i0[var] + self._var_to_len[var] - self.include_bnd[var][1]
        return slice(imin, imax)

    def all(self, var: str) -> slice:
        imin = self._var_to_i0[var]
        imax = self._var_to_i0[var] + self._var_to_len[var]
        return slice(imin, imax)

    def local_all(self, var: str) -> slice:
        imin = 1 - self.include_bnd[var][0]
        imax = self.nnodes - 1 + self.include_bnd[var][1]
        return slice(imin, imax)

    def itop(self, var: str) -> int:
        if not self.include_bnd[var][0]:
            raise RuntimeError(f"top boundary of {var} is not included")
        return self._var_to_i0[var]

    def ibot(self, var: str) -> int:
        if not self.include_bnd[var][1]:
            raise RuntimeError(f"bot boundary of {var} is not included")
        return self._var_to_i0[var] + self._var_to_len[var] - 1


@dataclass(frozen=True)
class Vector:
    slices: Slices
    arr: NDArray

    def __post_init__(self) -> None:
        assert self.arr.shape == (self.slices.total_size,)

    def extract(self, var: str) -> NDArray:
        values = np.zeros(self.slices.nnodes, dtype=self.arr.dtype)
        values[self.slices.local_all(var)] = self.arr[self.slices.all(var)]
        return values

    def normalize_by(self, norm: complex) -> Vector:
        """Normalize by given value."""
        return Vector(self.slices, self.arr / norm)

    def normalize_by_max_of(self, var: str) -> Vector:
        """Normalize by the value of var with maximum absolute value."""
        var_arr = self.extract(var)
        return self.normalize_by(var_arr[np.argmax(np.abs(var_arr))])


@dataclass(frozen=True)
class Matrix:
    slices: Slices
    dtype: type = np.float64

    @cached_property
    def _mat(self) -> NDArray:
        return np.zeros(
            (self.slices.total_size, self.slices.total_size),
            dtype=self.dtype,
        )

    def add_top(self, rowvar: str, colvar: str, values: NDArray) -> None:
        g_row = self.slices.itop(rowvar)
        g_cols = self.slices.all(colvar)
        l_cols = self.slices.local_all(colvar)
        self._mat[g_row, g_cols] += values[l_cols]

    def add_bot(self, rowvar: str, colvar: str, values: NDArray) -> None:
        g_row = self.slices.ibot(rowvar)
        g_cols = self.slices.all(colvar)
        l_cols = self.slices.local_all(colvar)
        self._mat[g_row, g_cols] += values[l_cols]

    def add_all(self, rowvar: str, colvar: str, values: NDArray) -> None:
        g_rows = self.slices.all(rowvar)
        l_rows = self.slices.local_all(rowvar)
        g_cols = self.slices.all(colvar)
        l_cols = self.slices.local_all(colvar)
        self._mat[g_rows, g_cols] += values[l_rows, l_cols]

    def add_bulk(self, rowvar: str, colvar: str, values: NDArray) -> None:
        g_rows = self.slices.bulk(rowvar)
        g_cols = self.slices.all(colvar)
        l_cols = self.slices.local_all(colvar)
        self._mat[g_rows, g_cols] += values[1:-1, l_cols]

    def full_mat(self) -> NDArray:
        return np.copy(self._mat)
