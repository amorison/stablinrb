#!/usr/bin/env python3
import pathlib
import typing

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection
import matplotlib.collections as mcol
from matplotlib.lines import Line2D
import numpy as np

from planets import EARTH, MOON, MARS
from misc import savefig


mpl.rcParams['font.size'] = 14


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
    return pnt.eta * pnt.d_crystal**2 / (pnt.delta_rho * pnt.g * pnt.h_crystal**3)


def stokes_time_th(pnt, thickness):
    """Compute thermal Stokes timescale."""
    pnt.h_crystal = thickness
    return pnt.eta * pnt.d_crystal**2 / (pnt.d_rho_t * pnt.g * pnt.h_crystal**3)


bodies = EARTH, MARS, MOON

cases_bcs = 'closed', r'$\Phi^+=10^{-2}$', r'$\Phi^+=\Phi^-=10^{-2}$'
cases_bulk = 'both', 'both_frozen', 'onlyTemp'
bcs_meta = {
    cases_bcs[0]:
        BndInfo('b', 'non-penetrative ($\Phi^+=\Phi^-=\infty$)'),
    cases_bcs[1]:
        BndInfo('g', 'flow-through at top ($\Phi^+=10^{-2}, \Phi^-=\infty$)'),
    cases_bcs[2]:
        BndInfo('r', 'flow-through at top and bottom ($\Phi^+=\Phi^-=10^{-2}$)'),
}

data_smo = {}
data = {
    cases_bulk[0]: {'linestyle': '-', 'linewidth': 2,
                    'label': 'Composition + Temperature + Moving Frame'},
    cases_bulk[1]: {'linestyle': ':', 'linewidth': 1,
                    'label': 'Composition + Temperature'},
    cases_bulk[2]: {'linestyle': '-.', 'linewidth': 1,
                    'label': 'Temperature + Moving Frame'},
    'onlyCompo': {'linestyle': '--', 'linewidth': 1,
                  'label': 'Composition + Moving Frame'},
}
data_part = {}

for body in bodies:
    data_smo[body.name] = {}
    data_part[body.name] = {}
    bodir = pathlib.Path(body.name)
    with h5py.File(bodir / 'CoolingSMO.h5', 'r') as h5f:
        data_smo[body.name]['thicknessSMO'] = h5f['thickness'][()]
        data_smo[body.name]['timeSMO'] = h5f['time'][()]
    with h5py.File(bodir / 'both' / 'interTime_D.h5', 'r') as h5f:
        for grp in h5f:
            data_part[body.name][grp] = {}
            data_part[body.name][grp]['h'] = h5f[grp]['h'][()]
            data_part[body.name][grp]['part_coef'] = h5f[grp]['part_coef'][()]
    for case in data:
        data[case][body.name] = {}
        data_case = data[case][body.name]
        with h5py.File(bodir / case / 'DestabTime.h5', 'r') as h5f:
            for grp in h5f:
                data_case[grp] = {}
                data_case[grp]['tau'] = h5f[grp]['tau'][()]
                data_case[grp]['thickness'] = h5f[grp]['thickness'][()]

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
                                  data_smo[body.name]['timeSMO'][-1] - data_smo[body.name]['timeSMO'],
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
    axis[iplot].annotate(body.name,
                         xy=(0.5, 0.05), xycoords='axes fraction',
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
               bbox_to_anchor=(-0.3, -0.7), loc="lower left", ncol=3)
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
            axis[ipl].loglog(stokes_case[tau<1e99] / 3.15e13,
                             tau[tau<1e99] / 3.15e13,
                             color=meta_bcs.color)
            st_min = stokes_case[tau<1e99][0] / 3.15e13
            st_max = stokes_case[-1] / 3.15e13
            fit_min = tau[tau<1e99][0] / 3.15e13
            fit_max = fit_min * (st_max / st_min)**fit_expnt
            axis[ipl].loglog((st_min, st_max), (fit_min, fit_max),
                             color=meta_bcs.color, linestyle='--', linewidth=1)
        axis[ipl].set_xlabel('Stokes time (Myr)')
        axis[ipl].annotate(body.name,
                           xy=(0.5, 0.05), xycoords='axes fraction',
                           fontsize=18, ha='center')
    axis[0].set_ylabel('LSA timescale (Myr)')
    plt.subplots_adjust(wspace=0.05)
    savefig(fig, 'destab_stokes_{}.pdf'.format(bulk_case))

fig, axis = plt.subplots(ncols=len(bodies)-1, figsize=(11, 4), sharey=True)
for iplot, body in enumerate(bodies[:-1]):
    for case_bcs, meta_bcs in bcs_meta.items():
        if case_bcs not in data['onlyTemp'][body.name]:
            continue
        data_temp = data['onlyTemp'][body.name][case_bcs]
        data_compo = data['onlyCompo'][body.name][case_bcs]
        axis[iplot].semilogy(
            data_compo['thickness'] / 1e3,
            data_compo['tau'] / data_temp['tau'],
            color=meta_bcs.color, label=meta_bcs.legend)
    axis[iplot].set_xlabel('Thickness of solid mantle (km)')
    axis[iplot].set_ylim(ymin=1e-2, ymax=5)
    axis[iplot].annotate(body.name,
                         xy=(0.5, 0.95), xycoords='axes fraction',
                         fontsize=18, ha='center', va='top')
axis[0].set_ylabel(r'$\tau_C/\tau_T$')
plt.subplots_adjust(wspace=0.05)
savefig(fig, 'Compo_Temp.pdf')

fig, axis = plt.subplots(nrows=len(bodies)-1, figsize=(6, 11), sharex=True)
for iplot, body in enumerate(bodies[:-1]):
    for case_bcs, meta_bcs in bcs_meta.items():
        if case_bcs not in data_part[body.name]:
            continue
        part_coef = data_part[body.name][case_bcs]['part_coef']
        hsolid = data_part[body.name][case_bcs]['h']
        # avoid faulty cases where the intersction is not found correctly
        slc = hsolid > 1e4
        axis[iplot].plot(
            part_coef[slc], hsolid[slc] / 1e3,
            color=meta_bcs.color)
    axis[iplot].set_ylabel('Thickness of solid mantle (km)')
    axis[iplot].annotate(body.name,
                         xy=(0.5, 0.15), xycoords='axes fraction',
                         fontsize=18, ha='center')
axis[-1].set_xlabel(r'Partition coefficient $D$')
axis[-1].set_xlim(left=0, right=1)
plt.subplots_adjust(hspace=0.05)

dummy_line = [[(0, 0)]]
llc = []
llbl = []
for case_bcs, meta_bcs in bcs_meta.items():
    llc.append(mcol.LineCollection(
        dummy_line,
        linestyles=['-'],
        linewidths=[2],
        colors=[meta_bcs.color]))
    llbl.append(meta_bcs.legend)
axis[-1].legend(llc, llbl,
                handler_map={type(llc[0]): HandlerDashedLines()},
                handleheight=2,
                bbox_to_anchor=(-0.2, -0.5), loc="lower left", ncol=1)
savefig(fig, 'part_coef.pdf')
