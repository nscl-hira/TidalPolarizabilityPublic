import operator
import cPickle as pickle
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
import tempfile
import TidalLove.TidalLove as tidal
from decimal import Decimal
import matplotlib.pyplot as plt
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
    df.fillna(0, inplace=True)

    constraints = pd.read_csv('Constraints/LowEnergySym.csv')

    chi_square = {}
    for index, row in df.iterrows():
        asym = sky.GetAsymEnergy(constraints['rho']*rho0, row)
        chi_square[index] = chisqr(asym, constraints['S'], constraints['S_Error'])
    sorted_model = sorted(chi_square.items(), key=operator.itemgetter(1))
    # only select the 50 best fit
    sorted_model = sorted_model[0:50]
    
    # only plot those best fit
    ax = plt.subplot(212)
    # plot background as comparison
    ax = utl.PlotSkyrmeEnergy(df, ax, color='b')
    ax = plt.subplot(211)
    df = df.loc[[name for (name, _) in sorted_model]]
    ax = utl.PlotSkyrmeSymEnergy(df, ax, color='r')
    ax.errorbar(constraints['rho'], constraints['S'], 
                xerr=constraints['rho_Error'], yerr=constraints['S_Error'], 
                fmt='o', color='w', ecolor='grey', 
                markersize=10, elinewidth=2)
    ax = plt.subplot(212)

    ax = utl.PlotSkyrmeEnergy(df, ax, color='r')
    value, contour = utl.GetContour(df, 0.3, 1)
    # write contour to file
    np.savetxt('Results/E_Constrainted_with_S.csv', np.array([contour, value]), delimiter=',') 
    ax.plot(value, contour, color='black')
    plt.show()
