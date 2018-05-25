from copy import copy
import itertools
color = itertools.cycle(('g', 'purple', 'r', 'black', 'grey', 'orange'))
linestyle = itertools.cycle(((15.,15.,2.,5.),(15.,15.),(3.,3.)))
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import pandas as pd

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky
import SelectFlow as sflow
import SelectAsym as sasym
import SelectPolarizability as spol
import SelectRadius as srad
from Utilities.Constants import *

if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_summary.csv', index_col=0)
    df.fillna(0, inplace=True)

    # Let's try to constrainted with Low energy points, then constraint with flow afterward
    LowDensityConstrainted, _ = sasym.SelectLowDensity('Constraints/LowEnergySym.csv', df)
    LDFlowSoft, patch_soft = sflow.SelectFlow('Constraints/FlowAsymSoft.csv', LowDensityConstrainted, 0.8,
                                              linewidth=5, edgecolor='black', alpha=1,
                                              hatch='/', lw=2, zorder=10, fill=False, label='Exp.+Asy_soft')
    LDFlowStiff, patch_stiff = sflow.SelectFlow('Constraints/FlowAsymStiff.csv', LowDensityConstrainted, 0.8,
                                                linewidth=5, edgecolor='red', alpha=1,
                                                hatch='\\', lw=2, zorder=10, fill=False, label='Exp.+Asy_stiff')
    JustFlowSoft, _ = sflow.SelectFlow('Constraints/FlowAsymSoft.csv', df, 0.8)
    JustFlowStiff, _ = sflow.SelectFlow('Constraints/FlowAsymStiff.csv', df, 0.8)
    PolPatchList = []
    interval = np.arange(12,14,0.5)
    for interval_min, interval_max in zip(interval[:-1], interval[1:]):
        patch ,_ = srad.SelectRadius('Results/Skyrme_summary.csv', df, interval_min, interval_max)
        PolPatchList.append(patch)

    interval = [(0, 400.), (400., 800.), (800., 1200)]
    Lambda_List = []
    for interval_min, interval_max in interval:
        _, sub_data = spol.SelectPolarizability('Results/Skryme_summary.csv', df, interval_min, interval_max)
        Lambda_List.append(sub_data)

    LowDensityConstrainted.to_csv('Results/LowDensityConstrainted.csv', sep=',')
    JustFlowSoft.to_csv('Results/FlowSoftAsym.csv', sep=',')
    JustFlowStiff.to_csv('Results/FlowStiffAsym.csv', sep=',')


    """
    We will try to summarize all the selections in a few plots
    Convert all the constraints to regions in Energy density region
    """
    value, contour = utl.GetContour(LowDensityConstrainted, 0.3, 1)
    _, patch_LD = sflow.ContourToPatches(value, contour, linewidth=4, edgecolor='black', facecolor='orange', alpha=.7, zorder=10, label='Low density constraints')
    value, contour = utl.GetContour(JustFlowSoft, 2, 4.5)
    _, patch_FSoft = sflow.ContourToPatches(value, contour, linewidth=4, edgecolor='black', facecolor='black', alpha=.7, zorder=10, label='Flow Asym soft constraints')
    value, contour = utl.GetContour(JustFlowStiff, 2, 4.5)
    _, patch_FStiff = sflow.ContourToPatches(value, contour, linewidth=4, edgecolor='black', facecolor='green', alpha=.7, zorder=10, label='Flow Asym stiff constraints')



    """
    Draw all the constraints and then draw all the selected Skryme
    """
    ax = plt.subplot(111)
    ax.add_patch(patch_LD)
    ax.add_patch(patch_FSoft)
    ax.add_patch(patch_FStiff)
    for patch in PolPatchList:
        patch.set_linestyle((20., linestyle.next()))
        patch.set_edgecolor(color.next())
        patch.set_fill(None)
        ax.add_patch(patch)

    ax.legend(loc='upper left', fontsize=20, numpoints=1)

    """
    range_ = [0, 5]
    # plot all the skyrmes
    utl.PlotSkyrmeEnergy(df, ax, color='b', range_=range_, pfrac=0.0)
    # plot skyrmes that are selected only by low density points
    utl.PlotSkyrmeEnergy(LowDensityConstrainted, ax, color='r', range_=range_, pfrac=0.0)
    # plot skyrmes that are selected both by low density points and soft asym from flow
    utl.PlotSkyrmeEnergy(LDFlowSoft, ax, color='black', range_=range_, pfrac=0.0)
    # plot skyrmes that are selected both by low density points and stiff asym from flow
    utl.PlotSkyrmeEnergy(LDFlowStiff, ax, color='green', range_=range_, pfrac=0.0)
    """

    ax.set_xlim([0, 5])
    ax.set_ylim([1, 1e3])
    ax.set_xlabel('$\\rho/\\rho_{0}$')
    ax.set_ylabel('E/A (MeV)')
    ax.set_yscale('log')

    plt.show()

    ax1, ax2 = utl.PlotMaster(df, Lambda_List, [r'%4.1f < $\Lambda$ < %4.1f' % (low, high) for low, high in interval])
    ax2.add_patch(patch_soft)
    ax2.add_patch(patch_stiff)
    ax2.legend(loc='lower right')
    plt.show()
   

    ax = plt.subplot(221)
    ax.plot(df['R(1.4)'], df['lambda(1.4)'], 'ro', color='b', label='All Skyrmes')
    ax.plot(LowDensityConstrainted['R(1.4)'], LowDensityConstrainted['lambda(1.4)'], 'ro', color='r', label='Low density constrainted')
    ax.set_xlabel('$R(1.4 M_{\\odot})$')
    ax.set_ylabel('$\\Lambda(1.4 M_{\\odot})$')
    ax.set_ylim([0, 2000])
    ax.legend(loc='upper left', fontsize=20, numpoints=1)
    
    ax = plt.subplot(222)
    ax.plot(df['R(1.4)'], df['lambda(1.4)'], 'ro', color='b', label='All Skyrmes')
    ax.plot(JustFlowSoft['R(1.4)'], JustFlowSoft['lambda(1.4)'], 'ro', color='r', label='Flow Soft Asym.')
    ax.set_xlabel('$R(1.4 M_{\\odot})$')
    ax.set_ylabel('$\\Lambda(1.4 M_{\\odot})$')
    ax.set_ylim([0, 2000])
    ax.legend(loc='upper left', fontsize=20, numpoints=1)
    
    ax = plt.subplot(223)
    ax.plot(df['R(1.4)'], df['lambda(1.4)'], 'ro', color='b', label='All Skyrmes')
    ax.plot(JustFlowStiff['R(1.4)'], JustFlowStiff['lambda(1.4)'], 'ro', color='r', label='Flow Stiff Asym.')
    ax.set_xlabel('$R(1.4 M_{\\odot})$')
    ax.set_ylabel('$\\Lambda(1.4 M_{\\odot})$')
    ax.set_ylim([0, 2000])
    ax.legend(loc='upper left', fontsize=20, numpoints=1)

    ax = plt.subplot(224)
    ax.plot(df['R(1.4)'], df['lambda(1.4)'], 'ro', color='b', label='All Skyrmes')
    ax.plot(LowDensityConstrainted['R(1.4)'], LowDensityConstrainted['lambda(1.4)'], 'ro', color='r', label='Low density constrainted')
    ax.plot(JustFlowSoft['R(1.4)'], JustFlowSoft['lambda(1.4)'], 'ro', color='g', label='Flow Soft Asym.')
    ax.plot(JustFlowStiff['R(1.4)'], JustFlowStiff['lambda(1.4)'], 'ro', color='orange', label='Flow Stiff Asym.')
    ax.set_xlabel('$R(1.4 M_{\\odot})$')
    ax.set_ylabel('$\\Lambda(1.4 M_{\\odot})$')
    ax.set_ylim([0, 2000])
    ax.legend(loc='upper left', fontsize=20, numpoints=1)
  
    plt.show()
   
