import itertools
color = itertools.cycle(('g', 'purple', 'r', 'black', 'orange')) 
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from Utilities.EOSCreator import EOSCreator

def power_law(x, a, b, c):
    return a*np.power(x, b) + c


def AddPressure(df):
    pressure = []
    for index, row in df.iterrows():
        if row['EOSType'] == '3Poly':
            eos = EOSCreator(row, **row).Get3Poly()[0]
            rho0 = 0.16
        else:
            eos = sky.Skryme(row)
            rho0 = eos.rho0
        pressure.append({'Model':index, 
                        'P(3rho0)':eos.GetPressure(3*rho0, 0),
                        'P(2rho0)':eos.GetPressure(2*rho0, 0), 
                        'P(1.5rho0)':eos.GetPressure(1.5*rho0, 0),
                        'P(rho0)':eos.GetPressure(rho0, 0),
                        'P(0.67rho0)':eos.GetPressure(0.67*rho0, 0),
                        'P_Sym(2rho0)':eos.GetPressure(2*rho0, 0.5), 
                        'P_Sym(1.5rho0)':eos.GetPressure(1.5*rho0, 0.5),
                        'P_Sym(rho0)':eos.GetPressure(rho0, 0.5),
                        'P_Sym(0.67rho0)':eos.GetPressure(0.67*rho0, 0.5),
                        'Sym(2rho0)':eos.GetAsymEnergy(2*rho0),
                        'Sym(1.5rho0)':eos.GetAsymEnergy(1.5*rho0),
                        'Sym(rho0)':eos.GetAsymEnergy(rho0),
                        'Sym(0.67rho0)':eos.GetAsymEnergy(0.67*rho0)})

        # try to convert the result to float if it returns an array of single element
        for key, val in pressure[-1].iteritems():
            try:
                val = np.asscalar(val)
                pressure[-1][key] = val
            except AttributeError:
                continue
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
        pressure = eos.GetPressure(2*rho0, 0)
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

    # drop outliners which is found by hand
    df.drop(['SkSC6', 'SkI1'], inplace=True)   

    ax = plt.subplot(122)
    df_with_p = AddPressure(df)
    ax.plot(df_with_p['lambda(1.4)'], df_with_p['P(2rho0)'], 'ro', label=None, marker='o', markerfacecolor='w', color='b')

    # fit a power law
    popt, pcov = curve_fit(power_law, df_with_p['lambda(1.4)'], df_with_p['P(2rho0)'], method='lm')
    xaxis = np.linspace(0, 1600, 100)
    ax.plot(xaxis, power_law(xaxis, *popt), label=r'$fit: %f\Lambda^{%f}+%f$' % tuple(popt))

    ax.legend(fontsize=25)
    ax.set_ylim([1e-2,3000])
    #ax.set_yscale('log')
    ax.set_xlabel(r'$Deformability\ \Lambda$')
    ax.set_ylabel(r'$P(2\rho_{0})\ (MeV/fm^{3})$')
    ax.set_xlim([60,1500])
    ax.set_xscale('log')
    ax.set_ylim([0, 60])

    ax = plt.subplot(121)
    ax.plot(df_with_p['lambda(1.4)'], df_with_p['Sym(2rho0)'], 'ro', marker='o', markerfacecolor='w', color='b')
    ax.set_ylim([1e-2,3000])
    #ax.set_yscale('log')
    ax.set_xlabel(r'$Deformability\ \Lambda$')
    ax.set_ylabel(r'$S(2\rho_{0})\ (MeV/fm^{3})$')
    ax.set_xlim([60,1500])
    ax.set_ylim([20, 60])
    ax.set_xscale('log')

    df_with_p.to_csv('2rho0.csv')
    plt.show()
