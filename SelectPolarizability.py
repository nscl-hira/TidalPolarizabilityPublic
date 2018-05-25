import itertools
color = itertools.cycle(('g', 'purple', 'r', 'black', 'grey', 'orange')) 
linestyle = itertools.cycle(('dashdot','dashed','dotted')) 
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import pandas as pd

import Utilities.Utilities as utl
from Utilities.Constants import *

   # obs -- observed value
    # exp -- expected value

def chisqr(obs, exp, error):
    return (np.power((obs-exp),2)/(np.power(error, 2))).sum()

def SelectPolarizability(constraint_filename, df, min_lambda=0, max_lambda=1600):
    #l_result = pd.read_csv(constraint_filename, index_col=0)
    # merge with polarizability results for each skryme
    #df = pd.concat([df, l_result], axis=1)

    sub_data = df.loc[(df['lambda(1.4)'] > min_lambda) & (df['lambda(1.4)'] < max_lambda)]
    if sub_data.shape[0] == 0:
        return None, None
    value, contour = utl.GetContour(sub_data, 0, 5)
    contour = [[x, y] for x, y in zip(value, contour)]
    path = pltPath.Path(contour)
    patch = patches.PathPatch(path, label='%5.1f < $\Lambda$ < %5.1f'%(min_lambda, max_lambda))
    return patch, sub_data


if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_summary.csv', index_col=0)
    df.fillna(0, inplace=True)

    # ax1 = plt.subplot(211)    
    ax2 = plt.subplot(111)
    interval = [(1,200), (200, 400), (400,800),(800,1600)]
    labels_PE = []
    labels_E = []
    text = []
    for interval_min, interval_max in interval:
        patch ,sub_data = SelectPolarizability('Results/Skyrme_summary.csv', df, interval_min, interval_max)
        col = color.next()
        labels_E.append(utl.PlotSkyrmeEnergy(sub_data, ax2, pfrac=0., color=col)[0])
        #labels_PE.append(utl.PlotSkyrmePressureEnergy(sub_data, ax1, range_=[0,5], pfrac=0., color=col)[0])
        text.append('$%s < \\Lambda < %s$' % (interval_min, interval_max))

    #ax1.legend(labels_PE, text, fontsize=30, loc='lower right')
    ax2.legend(labels_E, text, loc='upper left')
    ax2.set_ylim([0,200])
    plt.show()

    df_low = pd.read_csv('SkyrmeParameters/SkyrmeConstraintedLowDensity.csv', index_col=0)
    df_low.fillna(0, inplace=True)


    _,df = SelectPolarizability('Results/Skyrme_summary.csv', df, 0, 2000)
    _,df_low = SelectPolarizability('Results/Skyrme_summary.csv', df_low, 0, 2000)

    ax1 = plt.subplot(111)
    ax1.plot(df['R(1.4)'], df['lambda(1.4)'], 'ro', label='All skyrmes', color='b')
    #ax1.plot(df_low['R(1.4)'], df_low['lambda(1.4)'], 'ro', label='Skyrmes consistent with low energy constraints', color='r')
    ax1.set_xlabel('$R(1.4 M_{\odot})$')
    ax1.set_ylabel('$\Lambda(1.4 M_{\odot})$')
    ax1.set_xlim([4, 16])
    #ax1.legend(fontsize=30)
    plt.show()
