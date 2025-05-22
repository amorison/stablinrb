from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

if typing.TYPE_CHECKING:
    from .geometry import SphOps


class Gravity(ABC):
    @abstractmethod
    def with_ra(self, ra: float, ops: SphOps) -> NDArray[np.number]: ...


@dataclass(frozen=True)
class ConstantGravity(Gravity):
    def with_ra(self, ra: float, ops: SphOps) -> NDArray[np.number]:
        return np.diag(ra / ops.phys_coord)


@dataclass(frozen=True)
class LinearGravity(Gravity):
    """Gravity acceleration varying linearly with radius."""

    def with_ra(self, ra: float, ops: SphOps) -> NDArray[np.number]:
        return ra * (1.0 - ops.gamma) * ops.identity
