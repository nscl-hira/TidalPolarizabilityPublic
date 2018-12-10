from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from copy import copy

from Utilities import Utilities as utl
from DrawAllEOS import ContourToPatches
from MakeSkyrmeFileBisection import LoadSkyrmeFile


def DrawLIGO(ax, **kwargs):
    # newest constraint from LIGO group
    y = [10.78245, 39.5678, 39.5678, 10.78245, 10.78245]
    x = [70, 70, 580, 580, 70]
    path, LIGO = ContourToPatches(x, y, **kwargs)
    return ax.add_patch(copy(LIGO))

def DrawSoftAsym(ax, **kwargs):
    #x = [20, 900, 900, 20, 20]
    x = [200, 450, 450, 200, 200]
    y = [15.4, 15.4, 21.3, 21.3, 15.4]
    path, SoftAsym = ContourToPatches(x, y, **kwargs)
    return ax.add_patch(copy(SoftAsym))

def DrawStiffAsym(ax, **kwargs):
    x = [450,700,700,450,450]
    y = [31.94, 31.94, 38.16, 38.16, 31.94]
    path, StiffAsym = ContourToPatches(x, y, **kwargs)
    return ax.add_patch(copy(StiffAsym))


def DrawOurData(df, ax, **kwargs):
    return ax.scatter(df['lambda(1.4)'], df['P(2rho0)'], **kwargs)

def DrawLowSym(df, ax, **kwargs):
    return ax.scatter(df['lambda(1.4)'], df['Sym(0.67rho0)'], **kwargs)


if __name__ == '__main__':
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(10, 8))

    # load Skyrmes
    df = LoadSkyrmeFile('Results/Orig_mm2.17.csv')

    # draw from bottom layer to the top
    #DrawLIGO(ax, color='aqua', hatch='xx', fill=False)
    #DrawLIGO(ax, edgecolor='b', fill=False, linewidth='3')
    DrawSoftAsym(ax[1], color='r', fill=True, alpha=1, label='soft')
    DrawStiffAsym(ax[1], color='b', fill=True, alpha=1, label='stiff')
    DrawOurData(df, ax[1], marker='o', color='magenta', facecolor='white', zorder=10, linewidth=3, s=300, label='Skyrme')
    DrawLowSym(df, ax[0], marker='o', color='magenta', facecolor='white', zorder=10, linewidth=3, s=300, label='Skyrme')
    #DrawOurData(df[df['AgreeLowDensity'] == True], ax, marker='o', color='black', facecolor='white', zorder=11, linewidth=3, s=300, label='')

    fig.subplots_adjust(hspace=0.)
    ax[1].text(700, 5, r'$\rho=2\rho_{0}$', fontsize=25)
    ax[0].text(700, 29, r'$0.67\rho_{0}$', fontsize=25)
    #ax.text(25, 17, r'soft', fontsize=25

    ax[0].set_ylim([20,30])
    ax[0].set_yticks([20,25,30])
    ax[0].set_ylabel(r'S(MeV)')

    ax[1].set_xlim([20, 900])
    ax[1].set_ylim([0, 50])
    #ax[1].set_xscale('log')
    ax[1].set_xlabel(r'Deformability $\Lambda$')
    ax[1].set_ylabel(r'  P $(MeV/fm^3)$', fontsize=27)
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    #ax[1].set_xticks([30, 100, 300])
    ax[1].set_yticks([0, 20, 40])
    ax[1].xaxis.set_ticks_position('both')
    ax[1].yaxis.set_ticks_position('both')
    ax[1].legend()
    plt.show()
