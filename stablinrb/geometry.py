from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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


@dataclass(frozen=True)
class Cartesian(Geometry):
    """Cartesian geometry."""

    def name_stem(self) -> str:
        return "cart"

    def bounds(self) -> tuple[float, float]:
        return (-0.5, 0.5)

    def is_spherical(self) -> bool:
        return False


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
