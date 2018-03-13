#!/usr/bin/env python3
import pathlib

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


bodies = EARTH, MARS, MOON

color_grp = {'closed': 'b',
             r'$\Phi^+=10^{-2}$': 'g',
             r'$\Phi^+=\Phi^-=10^{-2}$': 'r'}
legend_grp = {
    'closed': 'closed ($\Phi^+=\Phi^-=\infty$)',
    r'$\Phi^+=10^{-2}$': 'open at top ($\Phi^+=10^{-2}, \Phi^-=\infty$)',
    r'$\Phi^+=\Phi^-=10^{-2}$': 'open at top and bottom ($\Phi^+=\Phi^-=10^{-2}$)'}

data_smo = {}
data = {
    'both': {'linestyle': '-', 'linewidth': 2,
             'label': 'Composition + Temperature + Moving Frame'},
    'both_frozen': {'linestyle': ':', 'linewidth': 1,
                    'label': 'Composition + Temperature'},
    'onlyTemp': {'linestyle': '-.', 'linewidth': 1,
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

fig, axis = plt.subplots(ncols=len(bodies), figsize=(16, 4), sharey=True)
for iplot, body in enumerate(bodies):
    axis[iplot].grid(axis='y', ls=':')
    thick_smo = data_smo[body.name]['thicknessSMO']
    cline, = axis[iplot].semilogy(thick_smo / 1e3,
                                  data_smo[body.name]['timeSMO'],
                                  color='k', label='Crystallization time')
    dline, = axis[iplot].semilogy(thick_smo / 1e3, thick_smo**2 / body.kappa,
                                  color='k', linestyle='--',
                                  label='Diffusive timescale')
    for case_bulk, data_bulk in data.items():
        for case_bcs in color_grp:
            if case_bcs in data_bulk[body.name]:
                case_data = data_bulk[body.name][case_bcs]
            else:
                continue
            if case_bulk == 'onlyTemp':
                case_data['tau'][case_data['tau'] < 0] = 1e99
            axis[iplot].semilogy(
                case_data['thickness'] / 1e3, case_data['tau'],
                ls=data_bulk['linestyle'], linewidth=data_bulk['linewidth'],
                color=color_grp[case_bcs], label=legend_grp[case_bcs])
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
    llc.append(mcol.LineCollection(dummy_line * 3,
                                   linestyles=[case_data['linestyle']]*3,
                                   linewidths=[case_data['linewidth']]*3,
                                   colors=['b', 'g', 'r']))
    llbl.append(case_data['label'])
for case_bcs, color in color_grp.items():
    llc.append(mcol.LineCollection(dummy_line * 3,
                                   linestyles=['-', ':', '-.'],
                                   linewidths=[2, 1, 1],
                                   colors=[color] * 3))
    llbl.append(legend_grp[case_bcs])
llc.extend([cline, dline])
llbl.extend(['Crystallization time', 'Diffusive timescale'])
axis[0].legend(llc, llbl,
               handler_map={type(llc[0]): HandlerDashedLines()},
               handleheight=2,
               bbox_to_anchor=(0.3, -0.5), loc="lower left", ncol=3)
plt.subplots_adjust(wspace=0.05)
savefig(fig, 'destabTimeAll.pdf')
