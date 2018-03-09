#!/usr/bin/env python3
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np

from planets import EARTH, MOON, MARS
from misc import savefig

bodies = EARTH, MARS, MOON

color_grp = {'closed': 'b',
             r'$\Phi^+=10^{-2}$': 'g',
             r'$\Phi^+=\Phi^-=10^{-2}$': 'r'}
legend_grp = {
    'closed': 'closed ($\Phi^\pm=\infty$)',
    r'$\Phi^+=10^{-2}$': 'open at top ($\Phi^+=10^{-2}$)',
    r'$\Phi^+=\Phi^-=10^{-2}$': 'open at top and bottom ($\Phi^\pm=10^{-2}$)'}

data_smo = {}
data = {}

for body in bodies:
    data_smo[body.name] = {}
    bodir = pathlib.Path(body.name)
    with h5py.File(bodir / 'CoolingSMO.h5', 'r') as h5f:
        data_smo[body.name]['thicknessSMO'] = h5f['thickness'].value
        data_smo[body.name]['timeSMO'] = h5f['time'].value
    data[body.name] = {}
    with h5py.File(bodir / 'both' / 'DestabTime.h5', 'r') as h5f:
        for grp in h5f:
            data[body.name][grp] = {}
            data[body.name][grp]['tau'] = h5f[grp]['tau'].value
            data[body.name][grp]['thickness'] = h5f[grp]['thickness'].value

fig, axis = plt.subplots(ncols=len(bodies), figsize=(16, 4), sharey=True)
for iplot, body in enumerate(bodies):
    axis[iplot].grid(axis='y', ls=':')
    thick_smo = data_smo[body.name]['thicknessSMO']
    axis[iplot].semilogy(thick_smo / 1e3, data_smo[body.name]['timeSMO'],
                         color='k', label='Crystallization time')
    axis[iplot].semilogy(thick_smo / 1e3, thick_smo**2 / body.kappa,
                         color='k', linestyle='--',
                         label='Diffusive timescale')
    for case_name in color_grp:
        if case_name in data[body.name]:
            case_data = data[body.name][case_name]
        else:
            continue
        axis[iplot].semilogy(case_data['thickness'] / 1e3, case_data['tau'],
                             color=color_grp[case_name],
                             label=legend_grp[case_name])
        axis[iplot].set_xlabel('Thickness of solid mantle (km)')
    axis[iplot].annotate(body.name, xy=(thick_smo[-1] / 2e3, 3e4),
                         fontsize=18, ha='center')
axis[0].set_ylim([1e4, 1e19])
axis[0].set_ylabel('Time')
axis[0].set_yticks([86400, 3.15e7, 3.15e10, 3.15e13, 3.15e16])
axis[0].set_yticklabels(['1 day', '1 year', '1 kyr', '1 Myr', '1 Gyr'])
axis[0].legend(bbox_to_anchor=(0.1, -0.3), loc="lower left", ncol=5)
savefig(fig, 'destabTimeAll.pdf')
