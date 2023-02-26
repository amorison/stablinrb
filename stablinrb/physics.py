from __future__ import annotations

import typing

import numpy as np
from scipy.optimize import brentq

if typing.TYPE_CHECKING:
    from typing import Callable, Optional, Union

    from numpy.typing import NDArray

    from .analyzer import Analyser


class PhysicalProblem:

    """Description of the physical problem"""

    def __init__(
        self,
        gamma: Optional[float] = None,
        h_int: float = 0,
        phi_top: Optional[float] = None,
        phi_bot: Optional[float] = None,
        C_top: Optional[float] = None,
        C_bot: Optional[float] = None,
        freeslip_top: bool = True,
        freeslip_bot: bool = True,
        heat_flux_top: Optional[float] = None,
        heat_flux_bot: Optional[float] = None,
        biot_top: Optional[float] = None,
        biot_bot: Optional[float] = None,
        lewis: Optional[float] = None,
        composition: Optional[Callable[[NDArray], NDArray]] = None,
        prandtl: Optional[float] = None,
        grad_ref_temperature: Union[str, Callable[[NDArray], NDArray]] = "conductive",
        eta_r: Optional[Callable[[NDArray], NDArray]] = None,
        cooling_smo: Optional[tuple[Callable, Callable]] = None,
        frozen_time: bool = False,
        ref_state_translation: bool = False,
        water: bool = False,
        thetar: float = 0,
    ):
        """Create a physical problem instance

        gamma is r_bot/r_top, cartesian if None

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
        self._observers: list[Analyser] = []
        self.gamma = gamma
        self.h_int = h_int
        self.phi_top = phi_top
        self.phi_bot = phi_bot
        self.C_top = C_top
        self.C_bot = C_bot
        self.freeslip_top = freeslip_top
        self.freeslip_bot = freeslip_bot
        self.heat_flux_top = heat_flux_top
        self.heat_flux_bot = heat_flux_bot
        self.biot_top = biot_top
        self.biot_bot = biot_bot
        self.lewis = lewis
        self.composition = composition
        self.prandtl = prandtl
        self.grad_ref_temperature = grad_ref_temperature
        self.eta_r = eta_r
        self.cooling_smo = cooling_smo
        self.frozen_time = frozen_time
        self.ref_state_translation = ref_state_translation
        # parameters for the stability of water cooled from below around 4C
        self.water = water
        self.thetar = thetar

    def bind_to(self, analyzer: Analyser) -> None:
        """Connect analyzer to physical problem

        The analyzer will be warned whenever the physical
        problem geometry has changed"""
        self._observers.append(analyzer)

    def name(self) -> str:
        """Construct a name for the current case"""
        name = []
        if self.spherical:
            name.append("sph")
            name.append(str(self.gamma).replace(".", "-"))
        else:
            name.append("cart")
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

    @property
    def gamma(self) -> Optional[float]:
        """Aspect ratio of spherical geometry"""
        return self._gamma

    @gamma.setter
    def gamma(self, value: Optional[float]) -> None:
        """Set spherical according to gamma"""
        self.spherical = (value is not None) and (0 < value < 1)
        self._gamma = value if self.spherical else None
        # warn bounded analyzers that the problem geometry has changed
        for analyzer in self._observers:
            analyzer.phys = self

    @property
    def heat_flux_bot(self) -> Optional[float]:
        """Imposed heat flux at bottom"""
        return self._hfbot

    @heat_flux_bot.setter
    def heat_flux_bot(self, value: Optional[float]) -> None:
        """Only one heat flux imposed at the same time"""
        if value is not None:
            self._hftop: Optional[float] = None
        self._hfbot = value

    @property
    def heat_flux_top(self) -> Optional[float]:
        """Imposed heat flux at top"""
        return self._hftop

    @heat_flux_top.setter
    def heat_flux_top(self, value: Optional[float]) -> None:
        """Only one heat flux imposed at the same time"""
        if value is not None:
            self._hfbot = None
        self._hftop = value

    @property
    def lewis(self) -> Optional[float]:
        """Lewis number"""
        return self._lewis

    @lewis.setter
    def lewis(self, value: Optional[float]) -> None:
        """Composition profile only if Lewis non finite"""
        if value is not None:
            self._composition: Optional[Callable[[NDArray], NDArray]] = None
        self._lewis = value

    @property
    def composition(self) -> Optional[Callable[[NDArray], NDArray]]:
        """Compositional reference profile"""
        return self._composition

    @composition.setter
    def composition(self, value: Optional[Callable[[NDArray], NDArray]]) -> None:
        """Composition profile only if Lewis non finite"""
        if value is not None:
            self._lewis = None
        self._composition = value


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
