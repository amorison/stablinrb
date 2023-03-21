import numpy as np
import pytest

import stablinrb.physics as phy
from stablinrb.cartesian import CartStability
from stablinrb.spherical import SphStability


def test_classic_rb_cart() -> None:
    ana = CartStability(
        chebyshev_degree=20,
        bc_mom_top=phy.FreeSlip(),
        bc_mom_bot=phy.FreeSlip(),
    )
    ra_th = 27 * np.pi**4 / 4
    harm_th = np.pi / np.sqrt(2)

    sigma = ana.growth_rate(harm_th, ra_th)
    assert np.isclose(sigma, 0.0)

    ra_c, harm_c = ana.critical_ra()
    assert np.isclose(ra_c, ra_th, rtol=1e-4)
    assert np.isclose(harm_c, harm_th, rtol=1e-2)


@pytest.mark.parametrize(
    "gamma,ra_th,harm_th", [(0.55, 711.95, 3), (0.99, 657.53, 221)]
)
def test_classic_rb_sph(gamma: float, ra_th: float, harm_th: float) -> None:
    ana = SphStability(
        chebyshev_degree=15,
        gamma=gamma,
    )

    ra_c, harm_c = ana.critical_ra()
    assert np.isclose(ra_c, ra_th)
    assert harm_c == harm_th


@pytest.mark.parametrize("phi", [1e-3, 3e-2, 1e-2])
def test_phi_cart(phi: float) -> None:
    ana = CartStability(
        chebyshev_degree=10,
        bc_mom_top=phy.PhaseChange(phi),
        bc_mom_bot=phy.PhaseChange(phi),
    )

    # these are only to leading order
    ra_th = 24 * phi - 81 / 256 * phi**2
    harm_th = 3 * np.sqrt(phi) / (4 * np.sqrt(2))

    ra_c, harm_c = ana.critical_ra()
    assert np.isclose(ra_c, ra_th, 1e-4)
    assert np.isclose(harm_c, harm_th, 5e-2)


@pytest.mark.parametrize("gamma", [0.2, 0.4, 0.6, 0.8])
@pytest.mark.parametrize("phi", [1e-3, 3e-2, 1e-2])
def test_phi_sph(gamma: float, phi: float) -> None:
    ana = SphStability(
        chebyshev_degree=10,
        gamma=gamma,
        bc_mom_top=phy.PhaseChange(phi),
        bc_mom_bot=phy.PhaseChange(phi),
    )

    ra_th = (
        24
        * (phi + gamma**2 * phi)
        * (1 - gamma**3)
        / (gamma * (1 - gamma) * (gamma**2 + 4 * gamma + 1))
    )
    harm_th = 1

    ra_c, harm_c = ana.critical_ra()
    assert np.isclose(ra_c, ra_th, 1e-2)
    assert harm_c == harm_th
