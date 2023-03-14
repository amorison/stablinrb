from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .geometry import CartRadOps
from .matrix import Bot, Bulk, Field, Matrix, Slices, Top

if typing.TYPE_CHECKING:
    from typing import Callable, Mapping, Optional, Sequence

    from numpy.typing import NDArray

    from .geometry import Geometry, Operators, RadialOperators
    from .matrix import VarSpec


class BoundaryCondition(ABC):
    @abstractmethod
    def add_top(self, var: str, mat: Matrix, operators: RadialOperators) -> float:
        ...

    @abstractmethod
    def add_bot(self, var: str, mat: Matrix, operators: RadialOperators) -> float:
        ...


@dataclass(frozen=True)
class Dirichlet(BoundaryCondition):
    value: float

    def add_top(self, var: str, mat: Matrix, operators: RadialOperators) -> float:
        mat.add_term(Top(var), operators.identity, var)
        return self.value

    def add_bot(self, var: str, mat: Matrix, operators: RadialOperators) -> float:
        mat.add_term(Bot(var), operators.identity, var)
        return self.value


@dataclass(frozen=True)
class Neumann(BoundaryCondition):
    value: float

    def add_top(self, var: str, mat: Matrix, operators: RadialOperators) -> float:
        mat.add_term(Top(var), operators.grad_r, var)
        return self.value

    def add_bot(self, var: str, mat: Matrix, operators: RadialOperators) -> float:
        mat.add_term(Bot(var), operators.grad_r, var)
        return self.value


@dataclass(frozen=True)
class Robin(BoundaryCondition):
    value: float
    biot: float

    def add_top(self, var: str, mat: Matrix, operators: RadialOperators) -> float:
        mat.add_term(Top(var), self.biot * operators.identity + operators.grad_r, var)
        return self.value

    def add_bot(self, var: str, mat: Matrix, operators: RadialOperators) -> float:
        mat.add_term(Bot(var), self.biot * operators.identity + operators.grad_r, var)
        return self.value


class ReferenceProfile(ABC):
    @abstractmethod
    def eval_with(self, operators: RadialOperators) -> NDArray:
        ...


@dataclass(frozen=True)
class DiffusiveProf(ReferenceProfile):
    bcs_top: BoundaryCondition
    bcs_bot: BoundaryCondition
    source: float = 0.0

    def eval_with(self, operators: RadialOperators) -> NDArray:
        mat = Matrix(
            slices=Slices(
                var_specs=(Field(var="f", include_top=True, include_bot=True),),
                nnodes=operators.phys_coord.size,
            ),
        )
        mat.add_term(Bulk("f"), operators.lapl_r, "f")
        rhs = np.full_like(operators.phys_coord, -self.source)
        rhs[0] = self.bcs_top.add_top("f", mat, operators)
        rhs[-1] = self.bcs_bot.add_bot("f", mat, operators)
        return np.linalg.solve(mat.array(), rhs)


@dataclass(frozen=True)
class ArbitraryProf(ReferenceProfile):
    ref_prof_from_coord: Callable[[NDArray], NDArray]

    def eval_with(self, operators: RadialOperators) -> NDArray:
        return self.ref_prof_from_coord(operators.phys_coord)


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
    c_0: Optional[float] = None

    def eval_with(self, operators: RadialOperators) -> NDArray:
        # only written in cartesian
        assert isinstance(operators, CartRadOps)
        c_s = (
            (1 - 1 / self.thick_tot**3) ** (1 - self.partition_coef)
            if self.c_0 is None
            else self.c_0
        )
        return c_s * (
            self.thick_tot**3
            / (self.thick_tot**3 - (operators.phys_coord + 1 / 2) ** 3)
        ) ** (1 - self.partition_coef)


class BCPerturbation(ABC):
    """Boundary condition for perturbations."""

    @abstractmethod
    def include(self) -> bool:
        ...

    @abstractmethod
    def add_top(self, var: str, mat: Matrix, operators: Operators) -> None:
        ...

    @abstractmethod
    def add_bot(self, var: str, mat: Matrix, operators: Operators) -> None:
        ...


@dataclass(frozen=True)
class Zero(BCPerturbation):
    def include(self) -> bool:
        return False

    def add_top(self, var: str, mat: Matrix, operators: Operators) -> None:
        pass

    def add_bot(self, var: str, mat: Matrix, operators: Operators) -> None:
        pass


@dataclass(frozen=True)
class ZeroFlux(BCPerturbation):
    def include(self) -> bool:
        return True

    def add_top(self, var: str, mat: Matrix, operators: Operators) -> None:
        mat.add_term(Top(var), operators.grad_r, var)

    def add_bot(self, var: str, mat: Matrix, operators: Operators) -> None:
        mat.add_term(Bot(var), operators.grad_r, var)


@dataclass(frozen=True)
class RobinPert(BCPerturbation):
    coef_var: float
    coef_grad: float

    def include(self) -> bool:
        return False

    def _opt(self, ops: Operators) -> NDArray:
        return self.coef_var * ops.identity + self.coef_grad * ops.grad_r

    def add_top(self, var: str, mat: Matrix, operators: Operators) -> None:
        mat.add_term(Top(var), self._opt(operators), var)

    def add_bot(self, var: str, mat: Matrix, operators: Operators) -> None:
        mat.add_term(Bot(var), self._opt(operators), var)


@dataclass(frozen=True)
class AdvDiffEq:
    bc_top: BCPerturbation
    bc_bot: BCPerturbation
    ref_prof: ReferenceProfile
    diff_coef: float = 1.0

    # FIXME: also handle lhs here
    def add_pert_eq(self, var: str, mat: Matrix, operators: Operators) -> None:
        self.bc_top.add_top(var, mat, operators)
        self.bc_bot.add_bot(var, mat, operators)
        mat.add_term(Bulk(var), self.diff_coef * operators.lapl, var)
        adv_tcond = operators.adv_r @ self.ref_prof.eval_with(operators.radial_ops)
        mat.add_term(Bulk(var), -np.diag(adv_tcond), operators.adv_vel_var)


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
    def include(self, geometry: Geometry) -> Mapping[str, bool]:
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

    def include(self, geometry: Geometry) -> Mapping[str, bool]:
        if geometry.is_spherical():
            return {"p": True, "q": True}
        return {"u": False, "w": False, "p": True}

    def add_top(self, mat: Matrix, ops: Operators) -> None:
        if ops.spherical:
            mat.add_term(Top("p"), ops.identity, "p")
            mat.add_term(Top("q"), ops.grad_r, "p")

    def add_bot(self, mat: Matrix, ops: Operators) -> None:
        if ops.spherical:
            mat.add_term(Bot("p"), ops.identity, "p")
            mat.add_term(Bot("q"), ops.grad_r, "p")


@dataclass(frozen=True)
class FreeSlip(BCMomentum):
    """Free-slip boundary: no tangential constraint."""

    @property
    def name(self) -> str:
        return "free"

    @property
    def flow_through(self) -> bool:
        return False

    def include(self, geometry: Geometry) -> Mapping[str, bool]:
        if geometry.is_spherical():
            return {"p": True, "q": True}
        return {"u": True, "w": False, "p": True}

    def add_top(self, mat: Matrix, ops: Operators) -> None:
        if ops.spherical:
            mat.add_term(Top("p"), ops.identity, "p")
            mat.add_term(Top("q"), ops.diff_r(2), "p")
        else:
            mat.add_term(Top("u"), ops.grad_r, "u")

    def add_bot(self, mat: Matrix, ops: Operators) -> None:
        if ops.spherical:
            mat.add_term(Bot("p"), ops.identity, "p")
            mat.add_term(Bot("q"), ops.diff_r(2), "p")
        else:
            mat.add_term(Bot("u"), ops.grad_r, "u")


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

    def include(self, geometry: Geometry) -> Mapping[str, bool]:
        if geometry.is_spherical():
            return {"p": True, "q": True}
        return {"u": True, "w": True, "p": True}

    def add_top(self, mat: Matrix, ops: Operators) -> None:
        # FIXME: rheology
        eta = ops.viscosity[0, 0]
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
        eta = ops.viscosity[-1, -1]
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


@dataclass(frozen=True)
class PhysicalProblem:
    """Description of the physical problem.

    prandtl: None if infinite
    water: to study convection in a layer of water cooled from below, around 4C.
    thetar: (T0-T1)/Delta T -1/2 with T0 the temperature of maximum density (4C),
       Delta T the total temperature difference across the layer and T1 the bottom T.
    """

    geometry: Geometry
    temperature: Optional[AdvDiffEq] = AdvDiffEq(
        bc_top=Zero(),
        bc_bot=Zero(),
        ref_prof=DiffusiveProf(bcs_top=Dirichlet(0.0), bcs_bot=Dirichlet(1.0)),
    )
    composition: Optional[AdvDiffEq] = None
    bc_mom_top: BCMomentum = FreeSlip()
    bc_mom_bot: BCMomentum = FreeSlip()
    prandtl: Optional[float] = None
    eta_r: Optional[Callable[[NDArray], NDArray]] = None
    cooling_smo: Optional[tuple[Callable, Callable]] = None
    frozen_time: bool = False
    ref_state_translation: bool = False
    water: bool = False
    thetar: float = 0.0

    @property
    def spherical(self) -> bool:
        return self.geometry.is_spherical()

    @property
    def domain_bounds(self) -> tuple[float, float]:
        """Boundaries of physical domain."""
        return (1.0, 2.0) if self.spherical else (-0.5, 0.5)

    def name(self) -> str:
        """Construct a name for the current case"""
        name = [
            self.geometry.name_stem(),
            "TOP",
            self.bc_mom_top.name,
            "BOT",
            self.bc_mom_bot.name,
        ]
        return "_".join(name).replace(".", "-")

    def var_specs(self) -> Sequence[VarSpec]:
        common = []
        if self.temperature is not None:
            common.append(
                Field(
                    var="T",
                    include_top=self.temperature.bc_top.include(),
                    include_bot=self.temperature.bc_bot.include(),
                )
            )
        if self.composition is not None:
            common.append(
                Field(
                    var="c",
                    include_top=self.composition.bc_top.include(),
                    include_bot=self.composition.bc_bot.include(),
                )
            )
        if self.spherical:
            inc_top = self.bc_mom_top.include(self.geometry)
            inc_bot = self.bc_mom_bot.include(self.geometry)
            return [
                # poloidal potential
                Field(var="p", include_top=inc_top["p"], include_bot=inc_bot["p"]),
                # lapl(poloidal)
                Field(var="q", include_top=inc_top["q"], include_bot=inc_bot["q"]),
                *common,
            ]
        # cartesian
        inc_top = self.bc_mom_top.include(self.geometry)
        inc_bot = self.bc_mom_bot.include(self.geometry)
        return [
            # pressure
            Field(var="p", include_top=inc_top["p"], include_bot=inc_bot["p"]),
            # velocities
            Field(var="u", include_top=inc_top["u"], include_bot=inc_bot["u"]),
            Field(var="w", include_top=inc_top["w"], include_bot=inc_bot["w"]),
            *common,
        ]


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


def visco_Arrhenius(eta_c: float, gamma: float) -> Callable[[NDArray], NDArray]:
    """Viscosity profile in a conductive shell"""
    # to be checked
    return lambda r: np.exp(
        np.log(eta_c) * gamma / (1 - gamma) * (1 - 1 / (r * (1 - gamma)))
    )
