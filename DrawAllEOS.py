from copy import copy
import numpy as np
import argparse
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
import matplotlib as mpl
import pandas as pd
#mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches

from Utilities.EOSDrawer import EOSDrawer
from Utilities.MakeMovie import CreateGif
from MakeSkyrmeFileBisection import LoadSkyrmeFile, CalculatePolarizability
from SelectPressure import AddPressure
from SelectAsym import SelectLowDensity
from SelectSpeedOfSound import AddCausailty
from Utilities.Constants import *
from Utilities.SkyrmeEOS import Skryme

def ContourToPatches(value, contour, **args):
    contour = [[x, y] for x, y in zip(value, contour)]
    path = pltPath.Path(contour)
    return path, patches.PathPatch(path, **args)


def DrawConstraints(ax):
    GW_constraints = pd.read_csv('Constraints/GWPressureConstraint.csv')
    path, GW_patch = ContourToPatches(GW_constraints['rho/rho0'], GW_constraints['P(MeV/fm3)'], zorder=10, alpha=0.5, color='aqua', facecolor='aqua', label='GW')

    ASoft = pd.read_csv('Constraints/FlowAsymSoft.csv')
    path, ASoft_patch = ContourToPatches(ASoft['rho/rho0'], ASoft['P(MeV/fm3)'], zorder=10, alpha=1, fill=False, color='r', label='flow')

    KaonSoft = pd.read_csv('Constraints/KaonSoft.csv')
    path, KaonSoft_patch = ContourToPatches(KaonSoft['rho/rho0'], KaonSoft['P(MeV/fm3)'], zorder=10, alpha=1, fill=False, color='r', linestyle='--', label='kaon')

    AStiff = pd.read_csv('Constraints/FlowAsymStiff.csv')
    path, AStiff_patch = ContourToPatches(AStiff['rho/rho0'], AStiff['P(MeV/fm3)'], zorder=10, alpha=1, fill=False, color='darkblue')
 
    KaonStiff = pd.read_csv('Constraints/KaonStiff.csv')
    path, KaonStiff_patch = ContourToPatches(KaonStiff['rho/rho0'], KaonStiff['P(MeV/fm3)'], zorder=10, alpha=1, fill=False, color='darkblue', linestyle='--')


    # plot the region and the legend
    ax.add_patch(copy(GW_patch))
    ax.add_patch(copy(ASoft_patch))
    ax.add_patch(copy(AStiff_patch))
    ax.add_patch(copy(KaonSoft_patch))
    ax.add_patch(copy(KaonStiff_patch))

    ax.text(4., 10., 'Soft', color='r')
    ax.text(0.8, 20, 'Stiff', color='darkblue')
    
    leg = ax.legend(loc='lower right', fontsize=15)
    leg.legendHandles[1].set_color('black')
    leg.legendHandles[2].set_color('black')
    ax.set_yscale('log')
    ax.set_xlabel(r'Density $(\rho/\rho_{0})$')
    ax.set_ylabel(r'Pressure $(MeV\ fm^{-3})$')
    ax.set_xlim([0, 6])
    ax.set_ylim([1e-2, 1e3])
    ax.set_yticks([0.1,1,10,100])


def DrawEoS(ax):
    df = LoadSkyrmeFile('Results/Orig_mm2.17.csv')
    df = df[df['NegSound']==False]

    drawer = EOSDrawer(df)
    drawer.DrawEOS(ax=ax, xname='rho/rho0', yname='GetPressure', labels=['Polytrope', 'Skyrme', 'Rel. gas', 'Crust'])
    GW_constraints = pd.read_csv('Constraints/GWPressureConstraint.csv')
    path, GW_patch = ContourToPatches(GW_constraints['rho/rho0'], GW_constraints['P(MeV/fm3)'], zorder=10, alpha=0.5, color='aqua', facecolor='aqua', label='GW')
    
    ax.add_patch(copy(GW_patch))

    ax.set_yscale('log')
    ax.set_xlabel(r'Density $(\rho/\rho_{0})$')
    ax.set_ylabel(r'Pressure $(MeV\ fm^{-3})$')
    ax.set_xlim([0, 6])
    ax.set_ylim([1e-2, 1e3])
    ax.set_yticks([0.1,1,10,100])

    ax.legend(loc='lower right', fontsize=15)


if __name__ == "__main__":


    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 8))
    DrawConstraints(ax[0])
    DrawEoS(ax[1])
    
    # empty out axis title
    for a in ax:
        a.set_ylabel('')
        a.set_xlabel('')
        
    fig.subplots_adjust(hspace=0.)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel(r'Density $(\rho/\rho_{0})$')
    plt.ylabel(r'Pressure $(MeV\ fm^{-3})$', labelpad=20)

    plt.show()
