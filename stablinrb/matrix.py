from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import ClassVar

import numpy as np

if typing.TYPE_CHECKING:
    from typing import Mapping, Optional, Sequence

    from numpy.typing import NDArray


class Position(ABC):
    collocated: ClassVar[bool]

    @abstractmethod
    def length(self, nnodes: int) -> int:
        """Number of points at this position."""


@dataclass(frozen=True)
class Top(Position):
    var: str
    collocated: ClassVar[bool] = True

    def length(self, nnodes: int) -> int:
        return 1


@dataclass(frozen=True)
class Bulk(Position):
    var: str
    collocated: ClassVar[bool] = True

    def length(self, nnodes: int) -> int:
        return nnodes - 2


@dataclass(frozen=True)
class Bot(Position):
    var: str
    collocated: ClassVar[bool] = True

    def length(self, nnodes: int) -> int:
        return 1


@dataclass(frozen=True)
class All(Position):
    var: str
    collocated: ClassVar[bool] = True

    def length(self, nnodes: int) -> int:
        return nnodes


@dataclass(frozen=True)
class Single(Position):
    var: str
    collocated: ClassVar[bool] = False

    def length(self, nnodes: int) -> int:
        return 1


class VarSpec(ABC):
    @abstractmethod
    def name(self) -> str:
        """Variable name."""

    @abstractmethod
    def length(self, nnodes: int) -> int:
        """Number of points for this variable."""

    @abstractmethod
    def elements(self) -> Sequence[Position]:
        """Elements over which the variable spans."""

    @abstractmethod
    def collocation(self, nnodes: int) -> Optional[slice]:
        """Slice of collocation nodes relevant for this variable."""


@dataclass(frozen=True)
class Field(VarSpec):
    var: str

    def name(self) -> str:
        return self.var

    def length(self, nnodes: int) -> int:
        return nnodes

    def elements(self) -> Sequence[Position]:
        return (Top(self.var), Bulk(self.var), Bot(self.var))

    def collocation(self, nnodes: int) -> slice:
        return slice(0, nnodes)


@dataclass(frozen=True)
class Scalar(VarSpec):
    var: str

    def name(self) -> str:
        return self.var

    def length(self, nnodes: int) -> int:
        return 1

    def elements(self) -> Sequence[Position]:
        return (Single(self.var),)

    def collocation(self, nnodes: int) -> None:
        return None


@dataclass(frozen=True)
class Slices:
    var_specs: Sequence[VarSpec]
    nnodes: int

    @cached_property
    def variables(self) -> Sequence[str]:
        """All the variable names."""
        return tuple(spec.name() for spec in self.var_specs)

    @cached_property
    def total_size(self) -> int:
        """Full number of points to describe all the variables."""
        return sum(spec.length(self.nnodes) for spec in self.var_specs)

    @cached_property
    def _spans(self) -> Mapping[Position, slice]:
        spans = {}
        cur_index = 0
        for elt in chain.from_iterable(spec.elements() for spec in self.var_specs):
            elt_len = elt.length(self.nnodes)
            assert elt_len > 0
            spans[elt] = slice(cur_index, cur_index + elt_len)
            cur_index += elt_len
        for spec in self.var_specs:
            var = spec.name()
            if Bulk(var) in spans:
                bslc = spans[Bulk(var)]
                spans[All(var)] = slice(
                    bslc.start - (Top(var) in spans),
                    bslc.stop + (Bot(var) in spans),
                )
        return spans

    @cached_property
    def _collocs(self) -> Mapping[Position, slice]:
        colloc: dict[Position, slice] = {}
        for spec in self.var_specs:
            if (col := spec.collocation(self.nnodes)) is not None:
                var = spec.name()
                colloc[All(var)] = col
                colloc[Bulk(var)] = slice(1, self.nnodes - 1)
                colloc[Top(var)] = slice(0, 1)
                colloc[Bot(var)] = slice(self.nnodes - 1, self.nnodes)
        return colloc

    def span(self, position: Position) -> slice:
        """Relevant indices in the global matrices/vectors."""
        return self._spans[position]

    def full_position(self, var: str) -> Position:
        """Position of all points of this variable."""
        if (pos := All(var)) in self._spans:
            return pos
        return Single(var)

    def collocation(self, position: Position) -> slice:
        """Relevant collocation nodes."""
        return self._collocs[position]


@dataclass(frozen=True)
class Vector:
    slices: Slices
    arr: NDArray

    def __post_init__(self) -> None:
        assert self.arr.shape == (self.slices.total_size,)

    def extract(self, var: str) -> NDArray:
        """Values of the variable, adding boundaries if needed."""
        pos = self.slices.full_position(var)
        if pos.collocated:
            values = np.zeros(self.slices.nnodes, dtype=self.arr.dtype)
            values[self.slices.collocation(pos)] = self.arr[self.slices.span(pos)]
            return values
        return self.arr[self.slices.span(pos)]

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

    def add_term(self, row: Position, operator: NDArray, var: str) -> None:
        """Add term to matrix.

        Operator is sliced along row and/or var (column) if the position
        in that direction is at collocation nodes.
        """
        col = self.slices.full_position(var)
        if row.collocated:
            operator = operator[self.slices.collocation(row)]
        if col.collocated:
            operator = operator[..., self.slices.collocation(col)]
        self._mat[self.slices.span(row), self.slices.span(col)] += operator

    def array(self) -> NDArray:
        return np.copy(self._mat)
