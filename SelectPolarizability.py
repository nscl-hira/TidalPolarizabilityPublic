import itertools
color = itertools.cycle(('g', 'purple', 'r', 'black', 'grey', 'orange')) 
linestyle = itertools.cycle(('-.','--',':')) 
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import pandas as pd

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *

   # obs -- observed value
    # exp -- expected value

def chisqr(obs, exp, error):
    return (np.power((obs-exp),2)/(np.power(error, 2))).sum()

if __name__ == "__main__":
    df = pd.read_csv('SkyrmeParameters/PawelSkyrme.csv', index_col=0)
    l_result = pd.read_csv('Results/Skyrme_summary.csv', index_col=0)
    df.fillna(0, inplace=True)
    # merge with polarizability results for each skryme
    df = pd.concat([df, l_result], axis=1)

    sub_data = []
    interval = 400
    min_lambda = 000
    max_lambda = 1600
    ax = plt.subplot(111)

    for val in xrange(min_lambda, max_lambda, interval):
        sub_data = df.loc[(df['lambda(1.4)'] > val) & (df['lambda(1.4)'] < val+interval)]
        if sub_data.shape[0] == 0:
            continue
        col = color.next()
        #utl.PlotSkyrmeEnergy(sub_data, ax, color=col)
        value, contour = utl.GetContour(sub_data, 0, 2)
        plt.plot(value, contour, zorder=3, color=col, linestyle=linestyle.next(), linewidth=4)
        contour = [[x, y] for x, y in zip(value, contour)]
        path = pltPath.Path(contour)
        patch = patches.PathPatch(path, facecolor=col, alpha=0., label='%5.1f < $\Lambda$ < %5.1f'%(val, val+interval))
        ax.add_patch(patch)

    #sub_data = df.loc[df['lambda(1.4)'] > max_lambda]
    #utl.PlotSkyrmeEnergy(sub_data, ax, color=color.next())

    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_xlim([0, 2])
    ax.set_ylabel('Energy (MeV/fm3)', fontsize=30)
    ax.set_ylim([0, 80])
    plt.legend(loc='upper left')
    plt.show()
