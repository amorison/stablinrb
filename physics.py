import numpy as np
from scipy.optimize import brentq, newton

class PhysicalProblem:

    """Description of the physical problem"""

    def __init__(self, gamma=None, h_int=0,
                 phi_top=None, phi_bot=None,
                 freeslip_top=True, freeslip_bot=True,
                 heat_flux_top=None, heat_flux_bot=None,
                 lewis=None, composition=None,
                 prandtl=None,
                 grad_ref_temperature=None,
                 eta_r=None, cooling_smo=None,
                 ref_state_translation=False,
                 water=False, thetar=0):
        """Create a physical problem instance

        gamma is r_bot/r_top, cartesian if None

        lewis: Lewis number if finite
        composition: reference compositional profile if Le->infinity

        Boundary conditions:
        phi_*: phase change number, no phase change if None
        freeslip_*: whether free-slip of rigid if no phase change
        heat_flux_*: heat flux, Dirichlet condition if None
        prandtl: None if infinite
        water: to study convection in a layer of water cooled from below, around 4C.
        thetar: (T0-T1)/Delta T -1/2 with T0 the temperature of maximum density (4C), 
           Delta T the total temperature difference across the layer and T1 the bottom T.
        """
        self._observers = []
        self.gamma = gamma
        self.h_int = h_int
        self.phi_top = phi_top
        self.phi_bot = phi_bot
        self.freeslip_top = freeslip_top
        self.freeslip_bot = freeslip_bot
        self.heat_flux_top = heat_flux_top
        self.heat_flux_bot = heat_flux_bot
        self.lewis = lewis
        self.composition = composition
        self.prandtl = prandtl
        self.grad_ref_temperature = grad_ref_temperature
        self.eta_r = eta_r
        self.cooling_smo = cooling_smo
        self.ref_state_translation = ref_state_translation
        # parameters for the stability of water cooled from below around 4C
        self.water = water
        self.thetar = thetar

    def bind_to(self, analyzer):
        """Connect analyzer to physical problem

        The analyzer will be warned whenever the physical
        problem geometry has changed"""
        self._observers.append(analyzer)

    def name(self):
        """Construct a name for the current case"""
        name = []
        if self.spherical:
            name.append('sph')
            name.append(np.str(self.gamma).replace('.', '-'))
        else:
            name.append('cart')
        if self.phi_top is not None:
            name.append('phiT')
            name.append(np.str(self.phi_top).replace('.', '-'))
        else:
            name.append(
                'freeT' if self.freeslip_top else 'rigidT')
        if self.phi_bot is not None:
            name.append('phiB')
            name.append(np.str(self.phi_bot).replace('.', '-'))
        else:
            name.append(
                'freeB' if self.freeslip_bot else 'rigidB')
        return '_'.join(name)

    @property
    def gamma(self):
        """Aspect ratio of spherical geometry"""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        """Set spherical according to gamma"""
        self.spherical = (value is not None) and (0 < value < 1)
        self._gamma = value if self.spherical else None
        # warn bounded analyzers that the problem geometry has changed
        for analyzer in self._observers:
            analyzer.phys = self

    @property
    def heat_flux_bot(self):
        """Imposed heat flux at bottom"""
        return self._hfbot

    @heat_flux_bot.setter
    def heat_flux_bot(self, value):
        """Only one heat flux imposed at the same time"""
        if value is not None:
            self._hftop = None
        self._hfbot = value

    @property
    def heat_flux_top(self):
        """Imposed heat flux at top"""
        return self._hftop

    @heat_flux_top.setter
    def heat_flux_top(self, value):
        """Only one heat flux imposed at the same time"""
        if value is not None:
            self._hfbot = None
        self._hftop = value

    @property
    def lewis(self):
        """Lewis number"""
        return self._lewis

    @lewis.setter
    def lewis(self, value):
        """Composition profile only if Lewis non finite"""
        if value is not None:
            self._composition = None
        self._lewis = value

    @property
    def composition(self):
        """Compositional reference profile"""
        return self._composition

    @composition.setter
    def composition(self, value):
        """Composition profile only if Lewis non finite"""
        if value is not None:
            self._lewis = None
        self._composition = value

def wtran(eps):
    """translation velocity as function of the reduced Rayleigh number"""
    if eps <= 0:
        wtr = 0
        wtrs = 0
        wtrl = 0
    else:
        # function whose roots are the translation velocity
        func = lambda wtra, eps: \
            wtra**2 * np.sinh(wtra / 2) - 6 * (1 + eps) * \
                (wtra * np.cosh(wtra / 2) - 2 * np.sinh(wtra / 2))
        dfunc = lambda wtra, eps: \
            0.5 * wtra * (wtra * np.cosh(wtra / 2) - 2 * (1 + 3 * eps) * np.sinh(wtra / 2))
        # value in the large Ra limit
        wtrl = 6 * (eps + 1)
        ful = func(wtrl, eps)
        # small Ra limit
        wtrs = 2 * np.sqrt(15 * eps)
        fus = func(wtrs, eps)
        if fus * ful > 0:
            # print('warning in wtran: ',eps, wtrs, fus, wtrl, ful)
            if np.abs(fus) < np.abs(ful):
                wguess = wtrs
            else:
                wguess = wtrl
            wtr = wguess
            # wtr = newton(func, wguess, fprime=dfunc, args=(eps))
            # if eps < 1.e-2:
                # wtr = wtrs
                # print('setting wtr = wtrs= ', wtrs)
        # complete solution
        else:
            wtr = brentq(func, wtrs, wtrl, args=(eps))
    return wtr, wtrs, wtrl


def compo_smo(thick_tot, partition_coef, c_0=None):
    """Composition profile

    Computed in the case of a rapidly crystalizing smo
    """
    # only written in cartesian
    if c_0 is None:
        c_0 = (1 - 1 / thick_tot**3)**(1 - partition_coef)
    return lambda z: c_0 * (thick_tot**3 / (thick_tot**3 -
        (z + 1 / 2)**3))**(1 - partition_coef)


def visco_Arrhenius(eta_c, gamma):
    """Viscosity profile in a conductive shell"""
    # to be checked
    return lambda r: np.exp(np.log(eta_c) * gamma / (1 - gamma) *
                            (1 - 1 / (r * (1-gamma))))
