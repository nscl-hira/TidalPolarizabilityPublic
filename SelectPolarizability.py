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

def SelectPolarizability(constraint_filename, df, interval=400, min_lambda=0, max_lambda=1600):
    l_result = pd.read_csv(constraint_filename, index_col=0)
    # merge with polarizability results for each skryme
    df = pd.concat([df, l_result], axis=1)
    patch_list = []

    for val in xrange(min_lambda, max_lambda, interval):
        sub_data = df.loc[(df['lambda(1.4)'] > val) & (df['lambda(1.4)'] < val+interval)]
        if sub_data.shape[0] == 0:
            continue
        value, contour = utl.GetContour(sub_data, 0, 5)
        contour = [[x, y] for x, y in zip(value, contour)]
        path = pltPath.Path(contour)
        patch = patches.PathPatch(path, label='%5.1f < $\Lambda$ < %5.1f'%(val, val+interval))
        patch_list.append(patch)
    return patch_list    


if __name__ == "__main__":
    df = pd.read_csv('SkyrmeParameters/PawelSkyrme.csv', index_col=0)
    df.fillna(0, inplace=True)
    patch_list = SelectPolarizability('Results/Skyrme_summary.csv', df)

    ax = plt.subplot(111)
   
    for patch in patch_list:
        patch.set_linestyle(linestyle.next())
        patch.set_edgecolor(color.next())
        patch.set_fill(None)
        ax.add_patch(patch)

    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_xlim([0, 2])
    ax.set_ylabel('Energy (MeV/fm3)', fontsize=30)
    ax.set_ylim([0, 80])
    plt.legend(loc='upper left')
    plt.show()
