from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np

if typing.TYPE_CHECKING:
    from dmsuite.poly_diff import DiffMatrices
    from numpy.typing import NDArray

    from .rheology import Rheology


class Operators(ABC):
    @property
    @abstractmethod
    def spherical(self) -> bool:
        """Whether geometry is spherical."""

    @property
    @abstractmethod
    def nodes(self) -> NDArray:
        """Position of nodes."""

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

    @abstractmethod
    def diff_r(self, order: int) -> NDArray:
        """Radial derivative."""

    @property
    @abstractmethod
    def diff_h(self) -> NDArray:
        """Horizontal derivative."""

    @property
    @abstractmethod
    def lapl_h(self) -> NDArray:
        """Horizontal laplacian."""

    @property
    @abstractmethod
    def lapl_r(self) -> NDArray:
        """Radial Laplacian."""

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

    @property
    @abstractmethod
    def viscosity(self) -> NDArray:
        """viscosity."""
        # FIXME: break the circular dependency between Rheology and Operators


@dataclass(frozen=True)
class CartOps(Operators):
    diff_mat: DiffMatrices
    wavenumber: float
    rheology: Rheology

    @property
    def spherical(self) -> bool:
        return False

    @property
    def nodes(self) -> NDArray:
        return self.diff_mat.nodes

    @property
    def phys_coord(self) -> NDArray:
        return self.diff_mat.nodes

    @property
    def identity(self) -> NDArray:
        return np.identity(self.nodes.size)

    @property
    def grad_r(self) -> NDArray:
        return self.diff_mat.at_order(1)

    def diff_r(self, order: int) -> NDArray:
        return self.diff_mat.at_order(order)

    @cached_property
    def diff_h(self) -> NDArray:
        return 1j * self.wavenumber * self.identity

    @cached_property
    def lapl_h(self) -> NDArray:
        return -(self.wavenumber**2) * self.identity

    @property
    def lapl_r(self) -> NDArray:
        return self.diff_mat.at_order(2)

    @property
    def lapl(self) -> NDArray:
        return self.lapl_r + self.lapl_h

    @property
    def adv_r(self) -> NDArray:
        return self.grad_r

    @property
    def adv_vel_var(self) -> str:
        return "w"

    @property
    def viscosity(self) -> NDArray:
        return self.rheology.viscosity(self)


@dataclass(frozen=True)
class SphOps(Operators):
    gamma: float
    diff_mat: DiffMatrices
    harm_degree: int
    rheology: Rheology

    @property
    def spherical(self) -> bool:
        return True

    @property
    def nodes(self) -> NDArray:
        return self.diff_mat.nodes

    @property
    def ell2(self) -> int:
        return self.harm_degree * (self.harm_degree + 1)

    @cached_property
    def lambda_(self) -> float:
        return (2 * self.gamma - 1) / (1 - self.gamma)

    @cached_property
    def phys_coord(self) -> NDArray:
        return self.nodes + self.lambda_

    @property
    def identity(self) -> NDArray:
        return np.identity(self.nodes.size)

    @property
    def grad_r(self) -> NDArray:
        return self.diff_mat.at_order(1)

    def diff_r(self, order: int) -> NDArray:
        return self.diff_mat.at_order(order)

    @property
    def diff_h(self) -> NDArray:
        raise RuntimeError("Horizontal derivative not defined in spherical.")

    @cached_property
    def lapl_h(self) -> NDArray:
        return -np.diag(self.ell2 / self.phys_coord**2)

    @cached_property
    def lapl_r(self) -> NDArray:
        two_o_rad = np.diag(2 / self.phys_coord)
        dr1 = self.diff_mat.at_order(1)
        return self.diff_mat.at_order(2) + two_o_rad @ dr1

    @property
    def lapl(self) -> NDArray:
        return self.lapl_r + self.lapl_h

    @property
    def adv_r(self) -> NDArray:
        # u_r = l(l+1)/r P
        ur_o_pol = np.diag(self.ell2 / self.phys_coord)
        return ur_o_pol @ self.grad_r

    @property
    def adv_vel_var(self) -> str:
        return "p"

    @property
    def viscosity(self) -> NDArray:
        return self.rheology.viscosity(self)
