from copy import copy
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import pandas as pd

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from SelectFlow import SelectFlow

def NumTrueAbovePercentage(list_, percentage):
    num_elements = float(len(list_))
    if float(np.count_nonzero(list_)/num_elements) > percentage:
        return True
    return False 

def ContourToPatches(value, contour, **args):
    contour = [[x, y] for x, y in zip(value, contour)]
    path = pltPath.Path(contour)
    return path, patches.PathPatch(path, **args)

def SelectFlowSym(constraint_filename, df, accept_percentage=0.8, xmin=2, xmax=4.5, **args):
    constraints = pd.read_csv(constraint_filename)
    path, patch = ContourToPatches(constraints['rho/rho0'], constraints['P(MeV/fm3)'], **args)

    inside_list = []
    n = np.linspace(xmin, xmax, 1000)
    for index, row in df.iterrows():
        eos = sky.Skryme(row)
        pressure = eos.GetAutoGradPressure(n*rho0, 0.5)
        inside = path.contains_points(np.array([n, pressure]).T)
        if NumTrueAbovePercentage(inside, accept_percentage):
            inside_list.append(index)
    df_selected = df.ix[inside_list]

    return df_selected, patch

if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_summary.csv', index_col=0)
    df.fillna(0, inplace=True)

    
    # load the constraints from flow experiments
    df_flow, patch = SelectFlowSym('Constraints/FlowSymMat.csv', df, 0.8,
                                linewidth=5, edgecolor='pink', alpha=1,
                                hatch='/', lw=2, zorder=10, fill=False, label='Exp.')
    # create cut fo stiff asym
    df_kaon, patch_kaon = SelectFlowSym('Constraints/KaonSymMat.csv', df, 0.8, xmin=1.22, xmax=2.2,
                                linewidth=5, edgecolor='navy', alpha=1,
                                hatch='\\', lw=2, zorder=10, fill=False, label='Kaon')

    # write result to file
    #df_soft.to_csv('SkyrmeParameters/SkyrmeConstraintedWithFlowSoft.csv', sep=',')
    #df_stiff.to_csv('SkyrmeParameters/SkyrmeConstraintedWithFlowStiff.csv', sep=',')

    ax1, ax2 = utl.PlotMaster(df, [df_kaon], ['Kaon'], ('b'), pfrac=0.5)

    # plot the region and the legend
    ax2.add_patch(copy(patch))
    ax2.add_patch(copy(patch_kaon))
    leg = ax2.legend(loc='lower right')

    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    plt.show()

    # See how good do the interpolation perform
    # Digitized curve are subjected to error
    # interpolation wiht spline may magnify those errors
    # Smoothing factor s is introduced to smooth the error
    # this section of the code will plot the result of smoothed interpolation
    # Because smoothing may overcompensate and distort the result
    additional_constraints_fig1 = pd.read_csv('Constraints/x_1_fig1.csv')
    eos_spline = sky.EOSSpline(np.linspace(1e-2, 3, 10), np.linspace(1e-2, 3, 10), 0, additional_constraints_fig1['rho/rho0']*rho0, additional_constraints_fig1['P(MeV/fm3)'], 0.5)
    plt.plot(additional_constraints_fig1['rho/rho0'], additional_constraints_fig1['P(MeV/fm3)'], 'ro', label='Digitized points')
    plt.plot(np.linspace(1e-2, 3, 500), eos_spline.GetAsymEnergy(np.linspace(1e-2, 3, 500)*rho0), label='Interpolated curve')
    plt.xlabel(r'$\rho/\rho_{0}$')
    plt.ylabel(r'$S(\rho)$')
    plt.legend()
    plt.show()

    ax1, ax2 = utl.PlotMaster(df, [df_kaon], ['Kaon'], ('b'), pfrac=0.)
    rho = np.linspace(1e-2, 3, 500)
    first = True
    for index, row in df_kaon.iterrows():
        Nuclear = sky.Skryme(row)
        if first:
            label = 'Skyrme \n+ Fig.1 Sym. \nEnergy Term'
            first = False
        else:
            label = None
        
        eos_spline = sky.EOSSpline(rho*rho0, Nuclear.GetEnergy(rho*rho0, pfrac=0.5), 0, additional_constraints_fig1['rho/rho0']*rho0, additional_constraints_fig1['P(MeV/fm3)'], 0.5)
        ax2.plot(rho, eos_spline.GetAutoGradPressure(rho*rho0, 0.0), color='black', linewidth=5, label=label)

    _, patch_soft = SelectFlow('Constraints/FlowAsymSoft.csv', df, 0.8,
                               linewidth=5, edgecolor='navy', alpha=1,
                               hatch='/', lw=2, zorder=10, fill=False, label='Exp.+Asy_soft')
    # create cut fo stiff asym
    _, patch_stiff = SelectFlow('Constraints/FlowAsymStiff.csv', df, 0.8,
                                linewidth=5, edgecolor='fuchsia', alpha=1,
                                hatch='\\', lw=2, zorder=10, fill=False, label='Exp.+Asy_stiff')
    # load the constraints from Kaon experiments
    _, patch_soft_kaon = SelectFlow('Constraints/KaonSoft.csv', df, 0.8, xmin=1.2, xmax=2.2,
                                    linewidth=5, edgecolor='cyan', alpha=1,
                                    hatch='\\', lw=2, zorder=10, fill=False, label='Kaon+Asy_soft')
    # create cut fo stiff asym
    _, patch_stiff_kaon = SelectFlow('Constraints/KaonStiff.csv', df, 0.8, xmin=1.2, xmax=2.2,
                                     linewidth=5, edgecolor='red', alpha=1,
                                     hatch='+', lw=2, zorder=10, fill=False, label='Kaon+Asy_stiff')
    # create cut fo fig. 1 asym
    _, patch_fig1_kaon = SelectFlow('Constraints/KaonFig1.csv', df, 0.8, xmin=1.2, xmax=2.2,
                                    linewidth=5, edgecolor='orange', alpha=1,
                                    hatch='/', lw=2, zorder=10, fill=False, label='Kaon+Asy_fig1')
    eos_spline = sky.EOSSpline(additional_constraints_fig1['rho/rho0']*rho0, additional_constraints_fig1['P(MeV/fm3)'], 0.5)
    print('P_digitized(0.67rho0): %f\nP_digitized(1.5rho0): %f\nP_digitized(2rho0): %f' % (eos_spline.GetAutoGradPressure(0.67*0.16, 0), eos_spline.GetAutoGradPressure(1.5*0.16, 0), eos_spline.GetAutoGradPressure(2*0.16, 0)))
    ax2.add_patch(patch_soft)
    ax2.add_patch(patch_stiff)
    ax2.add_patch(patch_soft_kaon)
    ax2.add_patch(patch_stiff_kaon)
    ax2.add_patch(patch_fig1_kaon)
    ax2.legend()

    plt.show()
