import itertools
color = itertools.cycle(('g', 'purple', 'r', 'black', 'orange')) 
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
    return path, patches.PathPatch(path)

def AddPressure(df):
    pressure = []
    for index, row in df.iterrows():
        eos = sky.Skryme(row)
        pressure.append({'Model':index, 
                        'P(2rho0)':eos.GetAutoGradPressure(2*rho0, 0), 
                        'P(1.5rho0)':eos.GetAutoGradPressure(1.5*rho0, 0),
                        'P(0.67rho0)':eos.GetAutoGradPressure(0.67*rho0, 0),
                        'P_Sym(2rho0)':eos.GetAutoGradPressure(2*rho0, 0.5), 
                        'P_Sym(1.5rho0)':eos.GetAutoGradPressure(1.5*rho0, 0.5),
                        'P_Sym(0.67rho0)':eos.GetAutoGradPressure(0.67*rho0, 0.5),
                        'Sym(2rho0)':eos.GetAsymEnergy(2*rho0),
                        'Sym(1.5rho0)':eos.GetAsymEnergy(1.5*rho0),
                        'Sym(0.67rho0)':eos.GetAsymEnergy(0.67*rho0)})
    data = pd.DataFrame.from_dict(pressure)
    data.set_index('Model', inplace=True)
    return pd.concat([df, data], axis=1)
    

def SelectPressure(df, p_min, p_max, **args):
    """
    Constraint EOS by selecting pressure at 2 rho0
    """
    inside_list = []
    for index, row in df.iterrows():
        eos = sky.Skryme(row)
        pressure = eos.GetAutoGradPressure(2*rho0, 0)
        if pressure > p_min and pressure < p_max:
            inside_list.append(index)
    df_selected = df.ix[inside_list]
    return df_selected

if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_pd_0.7.csv', index_col=0)
    df.fillna(0, inplace=True)

    
    # plot the region and the legend
    ax = plt.subplot(111)

    intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 7000)]

    for interval in intervals:
        df_part = SelectPressure(df, interval[0], interval[1])
        ax.plot(df_part['R(1.4)'], df_part['lambda(1.4)'], 'ro', marker='o', color=color.next(), label='%4.1f < $P(2\\rho_{0})$ < %4.1f $MeV/fm^{3}$' % (interval[0], interval[1]))
    ax.set_ylim([0,3000])
    ax.set_ylabel('lambda(1.4)')
    ax.set_xlabel('R(1.4)')
    plt.legend(loc='upper left')
    plt.show()

    
    ax = plt.subplot(122)
    df_with_p = AddPressure(df)
    ax.plot(df_with_p['lambda(1.4)'], df_with_p['P(0.67rho0)'], 'ro', marker='o', markerfacecolor='w', color='b')
    ax.set_ylim([1e-2,3000])
    #ax.set_yscale('log')
    ax.set_xlabel(r'$Deformability\ \Lambda$')
    ax.set_ylabel(r'$P(0.67\rho_{0})\ (MeV/fm^{3})$')
    ax.set_xlim([60,1500])
    ax.set_xscale('log')
    ax.set_ylim([0, 25])

    ax = plt.subplot(121)
    ax.plot(df_with_p['lambda(1.4)'], df_with_p['Sym(0.67rho0)'], 'ro', marker='o', markerfacecolor='w', color='b')
    ax.set_ylim([1e-2,3000])
    #ax.set_yscale('log')
    ax.set_xlabel(r'$Deformability\ \Lambda$')
    ax.set_ylabel(r'$S(0.67\rho_{0})\ (MeV/fm^{3})$')
    ax.set_xlim([60,1500])
    ax.set_ylim([20, 60])
    ax.set_xscale('log')

    #df_with_p.to_csv('2rho0.csv')
    plt.show()
