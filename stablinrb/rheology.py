from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from .geometry import Operators
    from .ref_prof import ReferenceProfile


class Rheology(ABC):
    @abstractmethod
    def constant(self) -> bool:
        ...

    @abstractmethod
    def viscosity(self, ops: Operators) -> NDArray:
        ...


@dataclass(frozen=True)
class Isoviscous(Rheology):
    def constant(self) -> bool:
        return True

    def viscosity(self, ops: Operators) -> NDArray:
        return np.ones_like(ops.nodes)


@dataclass(frozen=True)
class Arrhenius(Rheology):
    visc_scale: float
    temp_scale: float
    temp_offset: float
    temperature: ReferenceProfile

    def constant(self) -> bool:
        return False

    def viscosity(self, ops: Operators) -> NDArray:
        temp = self.temperature.eval_with(ops)
        return self.visc_scale * np.exp(self.temp_scale / (temp + self.temp_offset))
