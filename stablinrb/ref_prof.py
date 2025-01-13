from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .geometry import CartOps
from .matrix import Bot, Bulk, Field, Matrix, Slices, Top

if typing.TYPE_CHECKING:
    from typing import Callable

    from numpy.typing import NDArray

    from .geometry import Operators


class BoundaryCondition(ABC):
    @abstractmethod
    def add_top(self, var: str, mat: Matrix, ops: Operators) -> float: ...

    @abstractmethod
    def add_bot(self, var: str, mat: Matrix, ops: Operators) -> float: ...


@dataclass(frozen=True)
class Dirichlet(BoundaryCondition):
    value: float

    def add_top(self, var: str, mat: Matrix, ops: Operators) -> float:
        mat.add_term(Top(var), ops.identity, var)
        return self.value

    def add_bot(self, var: str, mat: Matrix, ops: Operators) -> float:
        mat.add_term(Bot(var), ops.identity, var)
        return self.value


@dataclass(frozen=True)
class Neumann(BoundaryCondition):
    value: float

    def add_top(self, var: str, mat: Matrix, ops: Operators) -> float:
        mat.add_term(Top(var), ops.grad_r, var)
        return self.value

    def add_bot(self, var: str, mat: Matrix, ops: Operators) -> float:
        mat.add_term(Bot(var), ops.grad_r, var)
        return self.value


@dataclass(frozen=True)
class Robin(BoundaryCondition):
    value: float
    biot: float

    def add_top(self, var: str, mat: Matrix, ops: Operators) -> float:
        mat.add_term(Top(var), self.biot * ops.identity + ops.grad_r, var)
        return self.value

    def add_bot(self, var: str, mat: Matrix, ops: Operators) -> float:
        mat.add_term(Bot(var), self.biot * ops.identity + ops.grad_r, var)
        return self.value


class ReferenceProfile(ABC):
    @abstractmethod
    def eval_with(self, ops: Operators) -> NDArray: ...


@dataclass(frozen=True)
class DiffusiveProf(ReferenceProfile):
    bcs_top: BoundaryCondition
    bcs_bot: BoundaryCondition
    source: float = 0.0

    def eval_with(self, ops: Operators) -> NDArray:
        mat = Matrix(
            slices=Slices(
                var_specs=(Field(var="f"),),
                nnodes=ops.phys_coord.size,
            ),
        )
        mat.add_term(Bulk("f"), ops.lapl_r, "f")
        rhs = np.full_like(ops.phys_coord, -self.source)
        rhs[0] = self.bcs_top.add_top("f", mat, ops)
        rhs[-1] = self.bcs_bot.add_bot("f", mat, ops)
        return np.linalg.solve(mat.array(), rhs)


@dataclass(frozen=True)
class ArbitraryProf(ReferenceProfile):
    ref_prof_from_coord: Callable[[NDArray], NDArray]

    def eval_with(self, ops: Operators) -> NDArray:
        return self.ref_prof_from_coord(ops.phys_coord)


@dataclass(frozen=True)
class FractionalCrystProf(ReferenceProfile):
    """Bottom-up fractional crystallization profile.

    This is useful for the compositional field.

    Attributes:
        thick_tot: total thickness of the magma ocean.
        partition_coef: partition coefficient.
        c_0: composition of first solid if set (otherwise equilibrium is assumed).
    """

    thick_tot: float
    partition_coef: float
    c_0: float | None = None

    def eval_with(self, ops: Operators) -> NDArray:
        # only written in cartesian
        assert isinstance(ops, CartOps)
        c_s = (
            (1 - 1 / self.thick_tot**3) ** (1 - self.partition_coef)
            if self.c_0 is None
            else self.c_0
        )
        return c_s * (
            self.thick_tot**3 / (self.thick_tot**3 - (ops.phys_coord + 1 / 2) ** 3)
        ) ** (1 - self.partition_coef)
