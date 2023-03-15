import numpy as np
from numpy.typing import NDArray

import stablinrb.geometry as geom
import stablinrb.physics as phy
from stablinrb.nonlin import NonLinearAnalyzer


def test_nonlin_rb() -> None:
    ana = NonLinearAnalyzer(
        phys=phy.PhysicalProblem(
            geometry=geom.Cartesian(),
        ),
        ncheb=10,
        nnonlin=2,
    )
    harm_c = ana.harm_c
    pik2 = np.pi**2 + harm_c**2
    pi_z = np.pi * ana.nodes

    theory = {
        (1, 1): {
            "u": -0.5 * np.sin(pi_z) * np.pi * 1j / harm_c,
            "w": 0.5 * np.cos(pi_z),
            "T": 0.5 * np.cos(pi_z) / pik2,
        },
        (2, 0): {
            "T": np.sin(2 * pi_z) / (16 * pik2 * np.pi),
        },
        (3, 1): {
            "u": -3j
            * (pik2**2)
            / harm_c
            * np.pi
            * np.sin(3 * pi_z)
            / 16
            / (pik2**3 - (9 * np.pi**2 + harm_c**2) ** 3),
            "w": pik2**2
            * np.cos(3 * pi_z)
            / 16
            / (pik2**3 - (9 * np.pi**2 + harm_c**2) ** 3),
        },
    }

    for (order, harm), profs in theory.items():
        mode_calc = ana.all_modes[ana.mode_index.index(order, harm)]
        for var, prof_th in profs.items():
            prof_calc = mode_calc.extract(var)
            rtol = 0.1 if order == 3 else 1e-5
            assert np.allclose(prof_calc, prof_th, rtol=rtol), f"{var}_{order}_{harm}"


def _w11(z: NDArray, phi: float) -> NDArray:
    """11 mode for the vertical velocity at low phi for both bc"""
    coefs = [
        0.5 * np.ones_like(z),
        -9 / 128 * z**2,
        9 * z**2 * (2.3 * z**2 - 2.2) / 1.6384e3,
        -0.5 * 2.7 * z**2 * (2.20 * z**2 - 3.09) / 2.097152e3,
        135 * z**2 * (-4.293 + 2.084 * z**2) / 4.294967296e6,
    ]
    return sum(
        (coef * phi**i for i, coef in enumerate(coefs)),
        start=np.zeros_like(z, dtype=np.complex128),
    )


def _u11(z: NDArray, phi: float) -> NDArray:
    """11 mode for the horizontal velocity at low phi for both bc"""
    coefs = [
        -3j / 8 * z,
        3j * z * (23 * z**2 - 11) / (512 * np.sqrt(2)),
        3j * (4.4 * z**2 - 3.09) * z / (2.62144e3 * np.sqrt(2)),
    ]
    return sum(
        (coef * phi ** (i + 0.5) for i, coef in enumerate(coefs)),
        start=np.zeros_like(z, dtype=np.complex128),
    )


def _p11(z: NDArray, phi: float) -> NDArray:
    """11 mode for the pressure at low phi for both bc"""
    coefs = [
        (39 / 32 - 2 * z**2) * z,
        9 * z * (35 * z**2 - 22) / 2048,
        -9 * z * (1.772 * z**2 - 0.751) / 4.194304e3,
    ]
    return sum(
        (coef * phi**i for i, coef in enumerate(coefs, start=1)),
        start=np.zeros_like(z, dtype=np.complex128),
    )


def _t11(z: NDArray, phi: float) -> NDArray:
    """11 mode for the temperature at low phi for both bc"""
    coefs = [
        (1 - 4 * z**2) / 16,
        -9 / 4096 * (1 - 4 * z**2),
        3 * (1 - 4 * z**2) * (212 * z**2 - 1) / 2.097152e6,
        27 * (1 - 4 * z**2) * (4.3 * z**2 + 1.07) / 2.68435456e6,
    ]
    return sum(
        (coef * phi**i for i, coef in enumerate(coefs)),
        start=np.zeros_like(z, dtype=np.complex128),
    )


def _t20(z: NDArray, phi: float) -> NDArray:
    """20 mode for the temperature at low phi for both bc"""
    coefs = [
        (1 - 4 * z**2) * z / 96,
        (-57 * z / 163840 + 3 * z**3 / 4096 + 27 * z**5 / 10240),
    ]
    return sum(
        (coef * phi**i for i, coef in enumerate(coefs)),
        start=np.zeros_like(z, dtype=np.complex128),
    )


def _p20(z: NDArray, phi: float) -> NDArray:
    """20 mode for the pressure at low phi for both bc"""
    return -0.25 * (1 / 16 - 0.5 * z**2 + z**4) * phi


def _t22(z: NDArray, phi: float) -> NDArray:
    """22 mode for the temperature at low phi for both bc"""
    zpol = (1 - 4 * z**2) * z
    coefs = [
        zpol / 96,
        -zpol * 35 / 49152,
        -zpol * 565 / 6291456,
    ]
    return sum(
        (coef * phi**i for i, coef in enumerate(coefs)),
        start=np.zeros_like(z, dtype=np.complex128),
    )


def _w22(z: NDArray, phi: float) -> NDArray:
    """22 mode for the vertical velocity at low phi for both bc"""
    coefs = [
        z / 128,
        (-1631 * z / 524288 - 17 * z**3 / 131072),
    ]
    return sum(
        (coef * phi**i for i, coef in enumerate(coefs, start=1)),
        start=np.zeros_like(z, dtype=np.complex128),
    )


def _u22(z: NDArray, phi: float) -> NDArray:
    """22 mode for the horizontal velocity at low phi for both bc"""
    coefs = [
        np.ones_like(z) * 1j / 96,
        -(1631 / 393216 + 17 * z**2 / 32768 + z**4 / 64) * 1j / np.sqrt(2),
    ]
    return sum(
        (coef * phi ** (i + 0.5) for i, coef in enumerate(coefs)),
        start=np.zeros_like(z, dtype=np.complex128),
    )


def test_nonlin_phase() -> None:
    phi = 1e-2
    ana = NonLinearAnalyzer(
        phys=phy.PhysicalProblem(
            geometry=geom.Cartesian(),
            bc_mom_top=phy.PhaseChange(phase_number=phi),
            bc_mom_bot=phy.PhaseChange(phase_number=phi),
        ),
        ncheb=20,
        nnonlin=2,
    )
    z = ana.nodes
    theory = {
        (1, 1): {
            # "u": _u11(z, phi),
            "w": _w11(z, phi),
            "T": _t11(z, phi),
            # "p": _p11(z, phi),
        },
        (2, 0): {
            "u": np.zeros_like(z),
            "w": np.zeros_like(z),
            "T": _t20(z, phi),
            # "p": _p20(z, phi),
        },
        (2, 2): {
            # "u": _u22(z, phi),
            # "w": _w22(z, phi),
            "T": _t22(z, phi),
        },
    }

    for (order, harm), profs in theory.items():
        mode_calc = ana.all_modes[ana.mode_index.index(order, harm)]
        for var, prof_th in profs.items():
            prof_calc = mode_calc.extract(var)
            assert np.allclose(prof_calc, prof_th, rtol=1e-3), f"{var}_{order}_{harm}"
