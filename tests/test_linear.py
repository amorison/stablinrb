import numpy as np

import stablinrb.geometry as geom
import stablinrb.physics as phy
from stablinrb.analyzer import LinearAnalyzer


def test_classic_rb_cart() -> None:
    ana = LinearAnalyzer(
        phys=phy.PhysicalProblem(
            geometry=geom.Cartesian(),
            bc_mom_top=phy.FreeSlip(),
            bc_mom_bot=phy.FreeSlip(),
        ),
        chebyshev_degree=20,
    )
    ra_th = 27 * np.pi**4 / 4
    harm_th = np.pi / np.sqrt(2)

    sigma, _ = ana.eigvec(harm_th, ra_th)
    assert np.isclose(sigma, 0.0)

    ra_c, harm_c = ana.critical_ra()
    assert np.isclose(ra_c, ra_th, rtol=1e-4)
    assert np.isclose(harm_c, harm_th, rtol=1e-2)
