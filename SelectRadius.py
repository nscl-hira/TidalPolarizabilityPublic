import itertools
color = itertools.cycle(('g', 'purple', 'r', 'black', 'grey', 'orange')) 
linestyle = itertools.cycle(('dashdot','dashed','dotted')) 
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

def SelectRadius(constraint_filename, df, min_radius=12, max_radius=14):
    l_result = pd.read_csv(constraint_filename, index_col=0)
    # merge with polarizability results for each skryme
    df = pd.concat([df, l_result], axis=1)

    sub_data = df.loc[(df['R(1.4)'] > min_radius) & (df['R(1.4)'] < max_radius)]
    if sub_data.shape[0] == 0:
        return None, None
    value, contour = utl.GetContour(sub_data, 0, 5)
    contour = [[x, y] for x, y in zip(value, contour)]
    path = pltPath.Path(contour)
    patch = patches.PathPatch(path, label='%5.1f km < $R(1.4 M_{\odot})$ < %5.1f km'%(min_radius, max_radius))
    return patch, sub_data.drop(list(l_result), axis=1) 


if __name__ == "__main__":
    df = pd.read_csv('SkyrmeParameters/PawelSkyrme.csv', index_col=0)
    df.fillna(0, inplace=True)

    ax = plt.subplot(111)    
    interval = np.arange(12,14,0.5)
    for interval_min, interval_max in zip(interval[:-1], interval[1:]):
        patch ,_ = SelectRadius('Results/Skyrme_summary.csv', df, interval_min, interval_max)
        if patch:
           patch.set_linestyle(linestyle.next())
           patch.set_edgecolor(color.next())
           patch.set_fill(None)
           ax.add_patch(patch)

    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_xlim([0, 2])
    ax.set_ylabel('E/A (MeV)', fontsize=30)
    ax.set_ylim([0, 80])
    plt.legend(loc='upper left', fontsize=30)
    plt.show()
