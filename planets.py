"""Planets parametrization"""
from types import SimpleNamespace
import inspect

import h5py
import numpy as np
import scipy.integrate as integ
import scipy.optimize as opt

STEFAN_BOLTZMANN = 5.67e-8


class Planet(SimpleNamespace):
    """Simple parametrization of a planet, default to Earth"""
    name = 'Earth'
    compo_effects = True
    dtime = 3e8
    # all in SI base units
    r_tot = 6371e3
    d_crystal = 2500e3
    t_crystal = 4500  # solidus temperature at d_crystal
    rho = 4e3
    g = 9.81
    alpha = 1e-5  # dV/dT /V
    beta = (3.3 - 4.4) / 3.3  # drho/dc /rho
    heat_capacity = 1e3  # in SMO
    latent_heat = 4e5  # J/kg
    emissivity = 1e-4
    temp_inf = 255
    kappa = 1e-6
    # composition
    part = 0.6  # partitioning coefficient
    c_feo_liq0 = 0.1  # part & c_feo_liq0 from Andrault et al, 2011
    dtmelt_dp = 2e-8
    dtmelt_dc = -700
    ra_smo = 1e30
    eta = 1e18

    def __init__(self, **kwargs):
        attributes = inspect.getmembers(Planet, lambda a:not(inspect.isroutine(a)))
        attributes = dict(a for a in attributes if a[0][0] != '_')
        attributes.update(kwargs)
        super().__init__(**attributes)
        self.h_crystal = 0

    @property
    def r_ext(self):
        """Height of solid"""
        return self.r_int + self.h_crystal

    @property
    def r_int(self):
        return self.r_tot - self.d_crystal

    @property
    def isen(self):
        return self.alpha * self.g / self.heat_capacity  # in SMO

    @property
    def c_0(self):
        """Composition of initial solid"""
        return self.part * self.c_feo_liq0

    @property
    def r_eut(self):
        """r at which c=1"""
        return (self.r_int**3 * self.c_0**(1 / (1-self.part)) +
                self.r_tot**3 * (1 - self.c_0**(1 / (1-self.part)))
               )**(1/3)

    @property
    def rayleigh(self):
        return (self.rho * self.g * self.alpha * self.delta_temp() *
                self.h_crystal**3 / (self.eta * self.kappa))

    @property
    def ra_comp(self):
        return (self.rho * self.g * self.beta * self.h_crystal**3 /
                (self.eta * self.kappa))

    @property
    def gamma(self):
        return self.r_int / self.r_ext

    @property
    def conductivity(self):
        return self.rho * self.heat_capacity * self.kappa

    @property
    def thick_smo(self):
        """Dimensionless SMO thickness"""
        return (self.r_tot - self.r_ext) / self.d_crystal

    @property
    def d_rho_t(self):
        """Density contrast due to superadiabatic temperature."""
        return self.alpha * self.delta_temp()

    @property
    def d_rho_c(self):
        """Density contrast due to composition."""
        return self.beta * (self.composition(self.r_int)
                            - self.composition(self.r_ext))

    @property
    def delta_rho(self):
        """Density contrast contributing to instability."""
        return self.d_rho_t + self.d_rho_c

    def tau(self, growth_rate):
        """Dimensional time from dimensionless growth rate"""
        return self.d_crystal**2 / (self.kappa * growth_rate)

    def radim(self, r):
        """Dimensional radius from dimensionless one"""
        return (r - 1) * self.h_crystal + self.r_int

    def composition(self, r):
        """Composition profile

        r is dimensionful radius
        """
        return (self.c_0 * ((self.r_tot**3 - self.r_int**3)
                            / (self.r_tot**3 - r**3))**(1-self.part)
                if r < self.r_eut else 1)

    def grad_temp(self, r, rm_isen=True):
        """Temperature gradient

        r is dimensionful radius
        Total if rm_isen is False,
        superadiabatic otherwise
        """
        total = -self.rho * self.g * self.dtmelt_dp
        if self.compo_effects and r < self.r_eut:
            total += self.dtmelt_dc * \
                (self.composition(r) * 3 * (1 - self.part) *
                 r**2 / (self.r_tot**3 - r**3))
        if rm_isen:
            total += self.t_crystal * self.isen * \
                np.exp(- self.isen * (r - self.r_int))
        return total

    def delta_temp(self, rm_isen=True):
        """temperature difference in solid part"""
        return -integ.quad(
            lambda r: self.grad_temp(r, rm_isen), self.r_int, self.r_ext)[0]

    @property
    def surf_temp(self):
        """Surface temperature determined by fixing
        the boundary layer Ra# at top of the SMO"""
        temp_bot_smo = self.t_crystal - self.delta_temp(False)
        temp_surf_pot = temp_bot_smo * np.exp(- self.isen *
                                              (self.r_tot - self.r_ext))
        tsurf_func = lambda ts: (self.emissivity * STEFAN_BOLTZMANN *
                                 (ts**4 - self.temp_inf**4) -
                                 self.conductivity *
                                 (temp_surf_pot - ts) * 0.16 *
                                 self.ra_smo**(2/7) * self.thick_smo**(-1/7) /
                                 self.d_crystal)
        return opt.fsolve(tsurf_func, self.temp_inf)[0]

    def cooling_time(self, h_max, outfile, verbose=False):
        """Compute time to evacuate latent heat and cool down SMO

        Based on grey body radiation and fixed boundary layer Ra
        """
        if outfile.exists():
            print('Reading cooling history from {}'.format(outfile))
            with h5py.File(outfile, 'r') as h5f:
                crystallized = h5f['thickness'].value
                time = h5f['time'].value
            return crystallized, time
        crystallized = [0]
        time = [0]
        dtime = self.dtime / 3e3  # to have first point earlier
        while crystallized[-1] < h_max:
            self.h_crystal = crystallized[-1]
            temp_top = self.t_crystal - self.delta_temp(False)
            gtemp_top = self.grad_temp(self.r_ext, False)
            heat_to_extract = self.rho * self.heat_capacity * \
                (gtemp_top + temp_top * self.isen)
            expis = np.exp(self.isen * (self.r_ext - self.r_tot))
            heat_to_extract *= (self.r_tot**2 * expis -
                                self.r_ext**2) / self.isen + \
                2 * (self.r_tot * expis - self.r_ext) / self.isen**2 + \
                2 * (expis - 1) / self.isen**3
            heat_to_extract += self.rho * self.latent_heat * self.r_ext**2 + \
                self.rho * self.heat_capacity * temp_top * self.r_ext**2
            gray_body = self.emissivity * STEFAN_BOLTZMANN * self.r_tot**2 * \
                (self.surf_temp**4 - self.temp_inf**4)
            drad = gray_body * dtime / heat_to_extract
            crystallized.append(self.h_crystal + drad)
            time.append(time[-1] + dtime)
            dtime = self.dtime
            if verbose and len(time)%1000==0:
                print(self.surf_temp, temp_top, crystallized[-1]/1e3, time[-1]/3.15e7)
        crystallized = np.array(crystallized)
        time = np.array(time)
        with h5py.File(outfile, 'w') as h5f:
            h5f['thickness'] = crystallized
            h5f['time'] = time
        return crystallized, time


EARTH = Planet()

MOON = Planet(
    name='Moon',
    dtime=3e6,
    d_crystal=1000e3,
    emissivity=1,
    g=1.62,
    ra_smo=1e30 * 1.62 / 9.81 * (1000e3 / 2500e3)**3,
    r_tot=1737e3,
    t_crystal=1500)

MARS = Planet(
    name='Mars',
    d_crystal=1300e3,
    emissivity=1e-3,
    temp_inf=212,
    g=3.71,
    ra_smo=1e30 * 3.71 / 9.81 * (1300e3 / 2500e3)**3,
    r_tot=3390e3,
    t_crystal=2400)
