from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .matrix import Bot, Bulk, Matrix, Top

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from .geometry import Operators
    from .ref_prof import ReferenceProfile


class BCPerturbation(ABC):
    """Boundary condition for perturbations."""

    @abstractmethod
    def add_top(self, var: str, mat: Matrix, ops: Operators) -> None:
        ...

    @abstractmethod
    def add_bot(self, var: str, mat: Matrix, ops: Operators) -> None:
        ...


@dataclass(frozen=True)
class Zero(BCPerturbation):
    def add_top(self, var: str, mat: Matrix, ops: Operators) -> None:
        mat.add_term(Top(var), ops.identity, var)

    def add_bot(self, var: str, mat: Matrix, ops: Operators) -> None:
        mat.add_term(Bot(var), ops.identity, var)


@dataclass(frozen=True)
class ZeroFlux(BCPerturbation):
    def add_top(self, var: str, mat: Matrix, ops: Operators) -> None:
        mat.add_term(Top(var), ops.grad_r, var)

    def add_bot(self, var: str, mat: Matrix, ops: Operators) -> None:
        mat.add_term(Bot(var), ops.grad_r, var)


@dataclass(frozen=True)
class RobinPert(BCPerturbation):
    coef_var: float
    coef_grad: float

    def _opt(self, ops: Operators) -> NDArray:
        return self.coef_var * ops.identity + self.coef_grad * ops.grad_r

    def add_top(self, var: str, mat: Matrix, ops: Operators) -> None:
        mat.add_term(Top(var), self._opt(ops), var)

    def add_bot(self, var: str, mat: Matrix, ops: Operators) -> None:
        mat.add_term(Bot(var), self._opt(ops), var)


@dataclass(frozen=True)
class AdvDiffEq:
    bc_top: BCPerturbation
    bc_bot: BCPerturbation
    ref_prof: ReferenceProfile
    diff_coef: float = 1.0

    # FIXME: also handle lhs here
    def add_pert_eq(self, var: str, mat: Matrix, ops: Operators) -> None:
        self.bc_top.add_top(var, mat, ops)
        self.bc_bot.add_bot(var, mat, ops)
        mat.add_term(Bulk(var), self.diff_coef * ops.lapl, var)
        adv_tcond = ops.adv_r @ self.ref_prof.eval_with(ops)
        mat.add_term(Bulk(var), -np.diag(adv_tcond), ops.adv_vel_var)


class BCMomentum(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def flow_through(self) -> bool:
        ...

    @abstractmethod
    def add_top(self, mat: Matrix, ops: Operators) -> None:
        ...

    @abstractmethod
    def add_bot(self, mat: Matrix, ops: Operators) -> None:
        ...


@dataclass(frozen=True)
class Rigid(BCMomentum):
    """Rigid boundary condition v=0."""

    @property
    def name(self) -> str:
        return "rigid"

    @property
    def flow_through(self) -> bool:
        return False

    def add_top(self, mat: Matrix, ops: Operators) -> None:
        if ops.spherical:
            mat.add_term(Top("p"), ops.identity, "p")
            mat.add_term(Top("q"), ops.grad_r, "p")
        else:
            mat.add_term(Top("u"), ops.identity, "u")
            mat.add_term(Top("w"), ops.identity, "w")

    def add_bot(self, mat: Matrix, ops: Operators) -> None:
        if ops.spherical:
            mat.add_term(Bot("p"), ops.identity, "p")
            mat.add_term(Bot("q"), ops.grad_r, "p")
        else:
            mat.add_term(Bot("u"), ops.identity, "u")
            mat.add_term(Bot("w"), ops.identity, "w")


@dataclass(frozen=True)
class FreeSlip(BCMomentum):
    """Free-slip boundary: no tangential constraint."""

    @property
    def name(self) -> str:
        return "free"

    @property
    def flow_through(self) -> bool:
        return False

    def add_top(self, mat: Matrix, ops: Operators) -> None:
        if ops.spherical:
            mat.add_term(Top("p"), ops.identity, "p")
            mat.add_term(Top("q"), ops.diff_r(2), "p")
        else:
            mat.add_term(Top("u"), ops.grad_r, "u")
            mat.add_term(Top("w"), ops.identity, "w")

    def add_bot(self, mat: Matrix, ops: Operators) -> None:
        if ops.spherical:
            mat.add_term(Bot("p"), ops.identity, "p")
            mat.add_term(Bot("q"), ops.diff_r(2), "p")
        else:
            mat.add_term(Bot("u"), ops.grad_r, "u")
            mat.add_term(Bot("w"), ops.identity, "w")


@dataclass(frozen=True)
class PhaseChange(BCMomentum):
    """Phase change boundary condition."""

    phase_number: float
    stress_jump: float = 0.0  # FIXME: stress_jump in cartesian

    @property
    def name(self) -> str:
        cbit = f"_C_{self.stress_jump}" if self.stress_jump != 0.0 else ""
        return f"phi_{self.phase_number}{cbit}"

    @property
    def flow_through(self) -> bool:
        return True

    def add_top(self, mat: Matrix, ops: Operators) -> None:
        # FIXME: rheology
        eta = ops.viscosity[0]
        if ops.spherical:
            rbnd = ops.phys_coord[0]
            mat.add_term(
                Top("p"),
                ops.diff_r(2)
                - ((1 + self.stress_jump) * ops.lapl_h + 2 / rbnd**2 * ops.identity),
                "p",
            )
            mat.add_term(
                Top("q"),
                (self.phase_number * rbnd - 2 * self.stress_jump) * (-ops.lapl_h)
                + 2 * eta * ops.lapl_h @ (ops.identity - rbnd * ops.grad_r),
                "p",
            )
            mat.add_term(Top("q"), -eta * (ops.identity + rbnd * ops.grad_r), "q")
        else:
            mat.add_term(Top("u"), ops.grad_r, "u")
            mat.add_term(Top("u"), ops.diff_h, "w")
            mat.add_term(Top("w"), -ops.identity, "p")
            mat.add_term(
                Top("w"),
                self.phase_number * ops.identity + 2 * eta * ops.grad_r,
                "w",
            )

    def add_bot(self, mat: Matrix, ops: Operators) -> None:
        # FIXME: rheology
        eta = ops.viscosity[-1]
        if ops.spherical:
            rbnd = ops.phys_coord[-1]
            mat.add_term(
                Bot("p"),
                ops.diff_r(2)
                - ((1 - self.stress_jump) * ops.lapl_h + 2 / rbnd**2 * ops.identity),
                "p",
            )
            mat.add_term(
                Bot("q"),
                (self.phase_number * rbnd - 2 * self.stress_jump) * ops.lapl_h
                + 2 * eta * ops.lapl_h @ (ops.identity - rbnd * ops.grad_r),
                "p",
            )
            mat.add_term(Bot("q"), -eta * (ops.identity + rbnd * ops.grad_r), "q")
        else:
            mat.add_term(Bot("u"), ops.grad_r, "u")
            mat.add_term(Bot("u"), ops.diff_h, "w")
            mat.add_term(Bot("w"), -ops.identity, "p")
            mat.add_term(
                Bot("w"),
                -self.phase_number * ops.identity + 2 * eta * ops.grad_r,
                "w",
            )


def wtran(eps: float) -> tuple[float, float, float]:
    """translation velocity as function of the reduced Rayleigh number

    Only relevant for finite phase-change number at both boundaries.
    Used to compute thestability of the translation solution wrt deforming
    modes. See Labrosse et al (JFM, 2018) for details.
    """
    if eps <= 0:
        wtr = 0.0
        wtrs = 0.0
        wtrl = 0.0
    else:

        def func(wtra: float, eps: float) -> float:
            """function whose roots are the translation velocity"""
            return wtra**2 * np.sinh(wtra / 2) - 6 * (1 + eps) * (
                wtra * np.cosh(wtra / 2) - 2 * np.sinh(wtra / 2)
            )

        # value in the large Ra limit
        wtrl = 6 * (eps + 1)
        ful = func(wtrl, eps)
        # small Ra limit
        wtrs = 2 * np.sqrt(15 * eps)
        fus = func(wtrs, eps)
        if fus * ful > 0:
            # Both approximate solutions must be close. Choose the
            # closest.
            if np.abs(fus) < np.abs(ful):
                wguess = wtrs
            else:
                wguess = wtrl
            wtr = wguess
        else:
            # Complete solution by bracketing (Brentq).
            wtr = brentq(func, wtrs, wtrl, args=(eps))
    return wtr, wtrs, wtrl
