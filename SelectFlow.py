import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import pandas as pd

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *

def NumTrueAbovePercentage(list_, percentage):
    num_elements = float(len(list_))
    if float(np.count_nonzero(list_)/num_elements > percentage):
        return True
    return False 

def ContourToPatches(value, contour, **args):
    contour = [[x, y] for x, y in zip(value, contour)]
    path = pltPath.Path(contour)
    return path, patches.PathPatch(path, **args)

def SelectFlow(constraint_filename, df, accept_percentage=0.8, **args):
    constraints = pd.read_csv(constraint_filename)
    path, patch = ContourToPatches(constraints['rho/rho0'], constraints['P(MeV/fm3)'], **args)

    inside_list = []
    n = np.linspace(2, 4.5, 1000)
    for index, row in df.iterrows():
        eos = sky.Skryme(row)
        pressure = eos.GetAutoGradPressure(n*rho0, 0)
        inside = path.contains_points(np.array([n, pressure]).T)
        if NumTrueAbovePercentage(inside, accept_percentage):
            inside_list.append(index)
    df_selected = df.ix[inside_list]

    return df_selected, patch

if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_summary.csv', index_col=0)
    df.fillna(0, inplace=True)

    
    # load the constraints from flow experiments
    df_soft, patch_soft = SelectFlow('Constraints/FlowAsymSoft.csv', df, 0.5, 
                                     linewidth=5, edgecolor='black', facecolor='black', alpha=.8,
                                     lw=2, zorder=10, label='Exp.+Asy_soft')
    # create cut fo stiff asym
    df_stiff, patch_stiff = SelectFlow('Constraints/FlowAsymStiff.csv', df, 0.5,
                                       linewidth=5, edgecolor='black', facecolor='pink', alpha=.8,
                                       lw=2, zorder=10, label='Exp.+Asy_stiff')

    # write result to file
    df_soft.to_csv('SkyrmeParameters/SkyrmeConstraintedWithFlowSoft.csv', sep=',')
    df_stiff.to_csv('SkyrmeParameters/SkyrmeConstraintedWithFlowStiff.csv', sep=',')

    # plot the region and the legend
    ax = plt.subplot(121)
    ax.add_patch(patch_soft)
    ax.add_patch(patch_stiff)
    ax.legend(loc='upper left')

    # Plot skyrme interaction for everything
    utl.PlotSkyrmePressure(df, ax, color='b', range_=[1.5,5], pfrac=0.)
    # Plot skyrme interaction after selection, plot onto the same graph for comparison
    utl.PlotSkyrmePressure(df_soft, ax, color='black', range_=[1.5,5], pfrac=0.)
    utl.PlotSkyrmePressure(df_stiff, ax, color='pink', range_=[1.5,5], pfrac=0.)
    

    # Plot pure neutron matter
    ax = plt.subplot(122)

    value, contour = utl.GetContour(df_soft, 2, 4.5)
    _, patch = ContourToPatches(value, contour, linewidth=5, 
                                edgecolor='black', facecolor='black', alpha=.8,
                                lw=2, zorder=10, label='Exp.+Asy_soft')
    ax.add_patch(patch)
    value, contour = utl.GetContour(df_stiff, 2, 4.5)
    _, patch = ContourToPatches(value, contour, linewidth=5, 
                                edgecolor='black', facecolor='pink', alpha=.8,
                                lw=2, zorder=11, label='Exp.+Asy_stiff')
    ax.add_patch(patch)
    ax.legend(loc='upper left')

    utl.PlotSkyrmeEnergy(df, ax, color='b', range_=[0,5], pfrac=0.0)
    utl.PlotSkyrmeEnergy(df_soft, ax, color='black', range_=[0,5], pfrac=0.0)
    utl.PlotSkyrmeEnergy(df_stiff, ax, color='pink', range_=[0,5], pfrac=0.0)
    #ax.set_ylim([1,500])
    ax.set_yscale('log')
    ax.set_ylabel('E/A for pure neutron matter (MeV)')
    plt.show()
