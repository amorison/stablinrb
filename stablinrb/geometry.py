from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np

if typing.TYPE_CHECKING:
    from dmsuite.poly_diff import DiffMatrices
    from numpy.typing import NDArray


class Geometry(ABC):
    @abstractmethod
    def name_stem(self) -> str:
        ...

    @abstractmethod
    def bounds(self) -> tuple[float, float]:
        ...

    @abstractmethod
    def is_spherical(self) -> bool:
        ...

    @abstractmethod
    def with_dmat(self, diff_mat: DiffMatrices) -> RadialOperators:
        ...


@dataclass(frozen=True)
class Cartesian(Geometry):
    """Cartesian geometry."""

    def name_stem(self) -> str:
        return "cart"

    def bounds(self) -> tuple[float, float]:
        return (-0.5, 0.5)

    def is_spherical(self) -> bool:
        return False

    def with_dmat(self, diff_mat: DiffMatrices) -> CartRadOps:
        return CartRadOps(diff_mat=diff_mat)


@dataclass(frozen=True)
class Spherical(Geometry):
    """Spherical geometry.

    Attributes:
        gamme: rbot / rtop
    """

    gamma: float

    def __post_init__(self) -> None:
        assert 0 < self.gamma < 1

    def name_stem(self) -> str:
        gam = str(self.gamma).replace(".", "-")
        return f"sph_{gam}"

    def bounds(self) -> tuple[float, float]:
        return (1.0, 2.0)

    def is_spherical(self) -> bool:
        return True

    def with_dmat(self, diff_mat: DiffMatrices) -> SphRadOps:
        return SphRadOps(geom=self, diff_mat=diff_mat)


class RadialOperators(ABC):
    @property
    @abstractmethod
    def phys_coord(self) -> NDArray:
        """Physical radial coordinate."""

    @property
    @abstractmethod
    def identity(self) -> NDArray:
        """Identity operator."""

    @property
    @abstractmethod
    def grad_r(self) -> NDArray:
        """Radial gradient."""

    @property
    @abstractmethod
    def lapl_r(self) -> NDArray:
        """Radial contribution to laplacian."""


@dataclass(frozen=True)
class CartRadOps(RadialOperators):
    diff_mat: DiffMatrices

    @property
    def nodes(self) -> NDArray:
        return self.diff_mat.nodes

    @property
    def phys_coord(self) -> NDArray:
        return self.nodes

    @cached_property
    def identity(self) -> NDArray:
        return np.identity(self.nodes.size)

    @property
    def grad_r(self) -> NDArray:
        return self.diff_mat.at_order(1)

    @property
    def lapl_r(self) -> NDArray:
        return self.diff_mat.at_order(2)


@dataclass(frozen=True)
class SphRadOps(RadialOperators):
    geom: Spherical
    diff_mat: DiffMatrices

    @property
    def nodes(self) -> NDArray:
        return self.diff_mat.nodes

    @cached_property
    def phys_coord(self) -> NDArray:
        return self.nodes + self.lambda_

    @cached_property
    def lambda_(self) -> float:
        return (2 * self.geom.gamma - 1) / (1 - self.geom.gamma)

    @cached_property
    def identity(self) -> NDArray:
        return np.identity(self.nodes.size)

    @property
    def grad_r(self) -> NDArray:
        return self.diff_mat.at_order(1)

    @cached_property
    def lapl_r(self) -> NDArray:
        two_o_rad = np.diag(2 / self.phys_coord)
        dr1 = self.diff_mat.at_order(1)
        return self.diff_mat.at_order(2) + two_o_rad @ dr1


class Operators(ABC):
    @property
    @abstractmethod
    def radial_ops(self) -> RadialOperators:
        """Radial operators."""

    @property
    @abstractmethod
    def phys_coord(self) -> NDArray:
        """Physical radial coordinate."""

    @property
    @abstractmethod
    def identity(self) -> NDArray:
        """Identity operator."""

    @property
    @abstractmethod
    def grad_r(self) -> NDArray:
        """Radial gradient."""

    @property
    @abstractmethod
    def lapl(self) -> NDArray:
        """Laplacian."""

    @property
    @abstractmethod
    def adv_r(self) -> NDArray:
        """Radial advection."""

    @property
    @abstractmethod
    def adv_vel_var(self) -> str:
        """Name of velocity variable to combine with `adv_r`."""


@dataclass(frozen=True)
class CartOps(Operators):
    rad_ops: CartRadOps
    wavenumber: float

    @property
    def radial_ops(self) -> CartRadOps:
        return self.rad_ops

    @property
    def phys_coord(self) -> NDArray:
        return self.rad_ops.phys_coord

    @property
    def identity(self) -> NDArray:
        return self.rad_ops.identity

    @property
    def grad_r(self) -> NDArray:
        return self.rad_ops.grad_r

    @property
    def lapl(self) -> NDArray:
        return self.rad_ops.lapl_r - self.wavenumber**2 * self.identity

    @property
    def adv_r(self) -> NDArray:
        return self.rad_ops.grad_r

    @property
    def adv_vel_var(self) -> str:
        return "w"


@dataclass(frozen=True)
class SphOps(Operators):
    rad_ops: SphRadOps
    harm_degree: int

    @property
    def radial_ops(self) -> SphRadOps:
        return self.rad_ops

    @property
    def ell2(self) -> int:
        return self.harm_degree * (self.harm_degree + 1)

    @property
    def phys_coord(self) -> NDArray:
        return self.rad_ops.phys_coord

    @property
    def identity(self) -> NDArray:
        return self.rad_ops.identity

    @property
    def grad_r(self) -> NDArray:
        return self.rad_ops.grad_r

    @property
    def lapl(self) -> NDArray:
        return self.rad_ops.lapl_r - np.diag(self.ell2 / self.phys_coord**2)

    @property
    def adv_r(self) -> NDArray:
        # u_r = l(l+1)/r P
        ur_o_pol = np.diag(self.ell2 / self.phys_coord)
        return ur_o_pol @ self.rad_ops.grad_r

    @property
    def adv_vel_var(self) -> str:
        return "p"
