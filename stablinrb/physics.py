from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .matrix import Bot, Bulk, Field, Matrix, Slices, Top

if typing.TYPE_CHECKING:
    from typing import Callable, Optional, Sequence, Union

    from numpy.typing import NDArray

    from .geometry import Geometry, RadialOperators
    from .matrix import VarSpec


# FIXME: difference between BCs for the reference and for the perturbation
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


@dataclass(frozen=True)
class DiffusiveField:
    bcs_top: BoundaryCondition
    bcs_bot: BoundaryCondition
    source: float = 0.0

    def ref_profile(self, operators: RadialOperators) -> NDArray:
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
class PhysicalProblem:
    """Description of the physical problem.

    grad_ref_temperature (spherical only): can be an arbitrary function of
        the radius. If set to 'conductive', the conductive temperature
        profile is used. If set to None, all temperature effects are
        switched off.
    lewis: Lewis number if finite
    composition: reference compositional profile if Le->infinity

    Boundary conditions:
    phi_*: phase change number, no phase change if None
    C_*: Chambat's parameter for phase change. Set to 0 if None
    freeslip_*: whether free-slip of rigid if no phase change
    heat_flux_*: heat flux, Dirichlet condition if None
    prandtl: None if infinite
    water: to study convection in a layer of water cooled from below, around 4C.
    thetar: (T0-T1)/Delta T -1/2 with T0 the temperature of maximum density (4C),
       Delta T the total temperature difference across the layer and T1 the bottom T.
    """

    geometry: Geometry
    h_int: float = 0.0
    phi_top: Optional[float] = None
    phi_bot: Optional[float] = None
    C_top: Optional[float] = None
    C_bot: Optional[float] = None
    freeslip_top: bool = True
    freeslip_bot: bool = True
    heat_flux_top: Optional[float] = None
    heat_flux_bot: Optional[float] = None
    biot_top: Optional[float] = None
    biot_bot: Optional[float] = None
    lewis: Optional[float] = None
    composition: Optional[Callable[[NDArray], NDArray]] = None
    prandtl: Optional[float] = None
    grad_ref_temperature: Union[str, Callable[[NDArray], NDArray]] = "conductive"
    eta_r: Optional[Callable[[NDArray], NDArray]] = None
    cooling_smo: Optional[tuple[Callable, Callable]] = None
    frozen_time: bool = False
    ref_state_translation: bool = False
    water: bool = False
    thetar: float = 0.0

    def __post_init__(self) -> None:
        # only one heat flux imposed at the same time
        assert self.heat_flux_bot is None or self.heat_flux_top is None
        # composition profile only at infinite Lewis
        assert not (self.composition and self.lewis is not None)

    @property
    def spherical(self) -> bool:
        return self.geometry.is_spherical()

    @property
    def domain_bounds(self) -> tuple[float, float]:
        """Boundaries of physical domain."""
        return (1.0, 2.0) if self.spherical else (-0.5, 0.5)

    def name(self) -> str:
        """Construct a name for the current case"""
        name = []
        name.append(self.geometry.name_stem())
        if self.phi_top is not None:
            name.append("phiT")
            name.append(str(self.phi_top).replace(".", "-"))
            if self.C_top is not None:
                name.append("CT")
                name.append(str(self.C_top).replace(".", "-"))
        else:
            name.append("freeT" if self.freeslip_top else "rigidT")
        if self.phi_bot is not None:
            name.append("phiB")
            name.append(str(self.phi_bot).replace(".", "-"))
            if self.C_bot is not None:
                name.append("CB")
                name.append(str(self.C_bot).replace(".", "-"))
        else:
            name.append("freeB" if self.freeslip_bot else "rigidB")
        return "_".join(name)

    def var_specs(self) -> Sequence[VarSpec]:
        common = [
            Field(
                var="T",
                include_top=self.heat_flux_top is not None or self.biot_top is not None,
                include_bot=self.heat_flux_bot is not None or self.biot_bot is not None,
            )  # temperature
        ]
        if self.composition is not None or self.lewis is not None:
            common.append(Field(var="c", include_top=False, include_bot=False))
        if self.spherical:
            return [
                # poloidal potential
                Field(var="p", include_top=True, include_bot=True),
                # lapl(poloidal)
                Field(var="q", include_top=True, include_bot=True),
                *common,
            ]
        # cartesian
        return [
            # pressure
            Field(var="p", include_top=True, include_bot=True),
            # velocities
            Field(
                var="u",
                include_top=self.phi_top is not None or self.freeslip_top,
                include_bot=self.phi_bot is not None or self.freeslip_bot,
            ),
            Field(
                var="w",
                include_top=self.phi_top is not None,
                include_bot=self.phi_bot is not None,
            ),
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


def compo_smo(
    thick_tot: float, partition_coef: float, c_0: Optional[float] = None
) -> Callable[[NDArray], NDArray]:
    """Composition profile

    Computed in the case of a rapidly crystalizing smo.
    """
    # only written in cartesian
    c_s = (1 - 1 / thick_tot**3) ** (1 - partition_coef) if c_0 is None else c_0
    return lambda z: c_s * (thick_tot**3 / (thick_tot**3 - (z + 1 / 2) ** 3)) ** (
        1 - partition_coef
    )


def visco_Arrhenius(eta_c: float, gamma: float) -> Callable[[NDArray], NDArray]:
    """Viscosity profile in a conductive shell"""
    # to be checked
    return lambda r: np.exp(
        np.log(eta_c) * gamma / (1 - gamma) * (1 - 1 / (r * (1 - gamma)))
    )
