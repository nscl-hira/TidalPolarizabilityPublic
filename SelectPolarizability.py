import operator
import cPickle as pickle
import itertools
color = itertools.cycle(('r', 'g', 'b', 'black', 'grey')) 
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
    # merge with polarizability results for each skryme
    df = pd.concat([df, l_result], axis=1)

    sub_data = []
    interval = 500
    max_lambda = 2000
    ax = plt.subplot(111)


    for val in xrange(0, max_lambda, interval):
        sub_data = df.loc[(df['lambda(1.4)'] > val) & (df['lambda(1.4)'] < val + interval)]
        col = color.next()
        #utl.PlotSkyrmeEnergy(sub_data, ax, color=col)
        value, contour = utl.GetContour(sub_data, 0, 2)
        ax.plot(value, contour, color=col, linewidth=3)
        
    #sub_data = df.loc[df['lambda(1.4)'] > max_lambda]
    #utl.PlotSkyrmeEnergy(sub_data, ax, color=color.next())


    plt.show()
