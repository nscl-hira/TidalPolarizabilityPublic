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
    l_result = pd.read_csv('Results/Skyrme_summary.csv', index_col=0)
    df.fillna(0, inplace=True)
    df = pd.concat([df, l_result], axis=1)
    df = df.loc[df['lambda(1.4)'] > 1000]
    #fig = plt.figure()
    utl.PlotSkyrmeEnergy(df)
    #ax.errorbar(constraints['rho'], constraints['S'], xerr=constraints['rho_Error'], yerr=constraints['S_Error'], fmt='o', color='w', ecolor='grey', markersize=10, elinewidth=2)
    plt.show()
