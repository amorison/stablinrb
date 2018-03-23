#!/usr/bin/env python3
import pathlib
import typing

import h5py
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection
import matplotlib.collections as mcol
from matplotlib.lines import Line2D
import numpy as np

from planets import EARTH, MOON, MARS
from misc import savefig


class HandlerDashedLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.

    https://matplotlib.org/examples/pylab_examples/legend_demo5.html
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = ((height) / (numlines + 1)) * np.ones(xdata.shape, float)
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[0] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines


class BndInfo(typing.NamedTuple):
    color: str
    legend: str


def stokes_time(pnt, thickness):
    """Compute Stokes timescale."""
    pnt.h_crystal = thickness
    return pnt.eta * pnt.thick_smo**2 / (pnt.delta_rho * pnt.g * pnt.h_crystal**3)


def stokes_time_th(pnt, thickness):
    """Compute thermal Stokes timescale."""
    pnt.h_crystal = thickness
    return pnt.eta * pnt.thick_smo**2 / (pnt.d_rho_t * pnt.g * pnt.h_crystal**3)


bodies = EARTH, MARS, MOON

cases_bcs = 'closed', r'$\Phi^+=10^{-2}$', r'$\Phi^+=\Phi^-=10^{-2}$'
cases_bulk = 'both', 'both_frozen', 'onlyTemp'
bcs_meta = {
    cases_bcs[0]:
        BndInfo('b', 'closed ($\Phi^+=\Phi^-=\infty$)'),
    cases_bcs[1]:
        BndInfo('g', 'open at top ($\Phi^+=10^{-2}, \Phi^-=\infty$)'),
    cases_bcs[2]:
        BndInfo('r', 'open at top and bottom ($\Phi^+=\Phi^-=10^{-2}$)'),
}

data_smo = {}
data = {
    cases_bulk[0]: {'linestyle': '-', 'linewidth': 2,
                    'label': 'Composition + Temperature + Moving Frame'},
    cases_bulk[1]: {'linestyle': ':', 'linewidth': 1,
                    'label': 'Composition + Temperature'},
    cases_bulk[2]: {'linestyle': '-.', 'linewidth': 1,
                    'label': 'Temperature + Moving Frame'},
}

for body in bodies:
    data_smo[body.name] = {}
    bodir = pathlib.Path(body.name)
    with h5py.File(bodir / 'CoolingSMO.h5', 'r') as h5f:
        data_smo[body.name]['thicknessSMO'] = h5f['thickness'].value
        data_smo[body.name]['timeSMO'] = h5f['time'].value
    for case in data:
        data[case][body.name] = {}
        data_case = data[case][body.name]
        with h5py.File(bodir / case / 'DestabTime.h5', 'r') as h5f:
            for grp in h5f:
                data_case[grp] = {}
                data_case[grp]['tau'] = h5f[grp]['tau'].value
                data_case[grp]['thickness'] = h5f[grp]['thickness'].value

stokes = {}
stokes_th = {}
for body in bodies:
    st_vec = np.vectorize(lambda t: stokes_time(body, t))
    st_vec_th = np.vectorize(lambda t: stokes_time_th(body, t))
    thick = data['both'][body.name]['closed']['thickness']
    stokes[body.name] = st_vec(thick)
    stokes_th[body.name] = st_vec_th(thick)

fig, axis = plt.subplots(ncols=len(bodies), figsize=(16, 4), sharey=True)
for iplot, body in enumerate(bodies):
    axis[iplot].grid(axis='y', ls=':')
    thick_smo = data_smo[body.name]['thicknessSMO']
    cline, = axis[iplot].semilogy(thick_smo / 1e3,
                                  data_smo[body.name]['timeSMO'],
                                  color='k', linewidth=1,
                                  label='Crystallization time')
    tsline, = axis[iplot].semilogy(
        data['both'][body.name]['closed']['thickness'] / 1e3,
        stokes[body.name], color='k', linestyle='--', linewidth=1)
    for case_bulk, data_bulk in data.items():
        for case_bcs, meta_bcs in bcs_meta.items():
            if case_bcs in data_bulk[body.name]:
                case_data = data_bulk[body.name][case_bcs]
            else:
                continue
            if case_bulk == 'onlyTemp':
                case_data['tau'][case_data['tau'] < 0] = 1e99
            axis[iplot].semilogy(
                case_data['thickness'] / 1e3, case_data['tau'],
                ls=data_bulk['linestyle'], linewidth=data_bulk['linewidth'],
                color=meta_bcs.color, label=meta_bcs.legend)
    axis[iplot].set_xlabel('Thickness of solid mantle (km)')
    axis[iplot].annotate(body.name, xy=(thick_smo[-1] / 2e3, 3e4),
                         fontsize=18, ha='center')
axis[0].set_ylim([1e4, 1e19])
axis[0].set_ylabel('Time')
axis[0].set_yticks([86400, 3.15e7, 3.15e10, 3.15e13, 3.15e16])
axis[0].set_yticklabels(['1 day', '1 year', '1 kyr', '1 Myr', '1 Gyr'])

dummy_line = [[(0, 0)]]
llc = []
llbl = []
for case_bulk, case_data in data.items():
    llc.append(mcol.LineCollection(
        dummy_line * 3, linestyles=[case_data['linestyle']]*3,
        linewidths=[case_data['linewidth']]*3,
        colors=[bcs_meta[b].color for b in cases_bcs]))
    llbl.append(case_data['label'])
for case_bcs, meta_bcs in bcs_meta.items():
    llc.append(mcol.LineCollection(
        dummy_line * 3,
        linestyles=[data[b]['linestyle'] for b in cases_bulk],
        linewidths=[data[b]['linewidth'] for b in cases_bulk],
        colors=[meta_bcs.color] * 3))
    llbl.append(meta_bcs.legend)
llc.extend([cline, tsline])
llbl.extend(['Crystallization time', 'Stokes time'])
axis[0].legend(llc, llbl,
               handler_map={type(llc[0]): HandlerDashedLines()},
               handleheight=2,
               bbox_to_anchor=(0.3, -0.5), loc="lower left", ncol=3)
plt.subplots_adjust(wspace=0.05)
savefig(fig, 'destabTimeAll.pdf')

for fit_expnt, bulk_case in zip((1, 1, 1), cases_bulk):
    fig, axis = plt.subplots(ncols=len(bodies) - (bulk_case == 'onlyTemp'),
                             figsize=(16, 4), sharey=True)
    for ipl, body in enumerate(bodies):
        if body is MOON and bulk_case == 'onlyTemp':
            continue
        case_data = data[bulk_case][body.name]
        stokes_case = (stokes[body.name] if bulk_case != 'onlyTemp'
                       else stokes_th[body.name])
        for case_bcs, meta_bcs in bcs_meta.items():
            if case_bcs not in case_data:
                continue
            tau = case_data[case_bcs]['tau']
            axis[ipl].loglog(stokes[body.name][tau<1e99], tau[tau<1e99],
                             color=meta_bcs.color)
            st_min = stokes[body.name][tau<1e99][0]
            st_max = stokes[body.name][-1]
            fit_min = tau[tau<1e99][0]
            fit_max = fit_min * (st_max / st_min)**fit_expnt
            axis[ipl].loglog((st_min, st_max), (fit_min, fit_max),
                             color=meta_bcs.color, linestyle='--', linewidth=1)
        axis[ipl].set_xlabel('Stokes time')
    axis[0].set_ylabel('LSA timescale')
    plt.subplots_adjust(wspace=0.05)
    savefig(fig, 'destab_stokes_{}.pdf'.format(bulk_case))
