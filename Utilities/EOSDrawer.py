import sys
from pathos.multiprocessing import ProcessingPool
import scipy.optimize as opt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import SkyrmeEOS as sky
from Constants import *
from EOSCreator import EOSCreator

from itertools import izip

class Test:

    def __init__(self, eos):
        self.eos = eos

def GetEOSCreator(name_and_row):
    name = name_and_row[0]
    row = name_and_row[1]
    eos = sky.Skryme(row)
    creator = EOSCreator(eos, **row) 
    test = Test(eos)
    print('finished %s' % name)
    return name, creator

class EOSDrawer:


    def __init__(self, df):
        self.df = df
        name_list = [(index, row) for index, row in df.iterrows()]
        result = ProcessingPool().map(GetEOSCreator, name_list)
        self.EOSCreator = {val[0]: val[1] for val in result}

    def DrawEOS(self, df=None, ax=plt.subplot(111), xname='GetEnergyDensity', yname='GetAutoGradPressure', xlim=None, ylim=None, color=['r', 'b', 'g', 'orange', 'b', 'pink'], **kwargs):

        rho = np.concatenate([np.logspace(np.log(1e-9), np.log(3.76e-4), 100, base=np.exp(1)), np.linspace(3.77e-4, 1.6, 900)])
        if df is None:
            df = self.df
        for index, row in df.iterrows():
            if 'rho0' in row:
                rho0 = row['rho0']
            else:
                rho0 = 0.16

            eos, trans_dens = self.EOSCreator[index].GetEOSType(row['EOSType'])
            trans_dens = [10*rho0] + trans_dens + [1e-9]
            for num, (low_den, high_den) in enumerate(zip(trans_dens, trans_dens[1:])):
                rho = np.linspace(low_den, high_den, 100)
                if xname == 'rho':
                    x = rho
                else:
                    x = getattr(eos, xname)(rho, 0)
                if yname == 'rho':
                    y = rho
                else:
                    y = getattr(eos, yname)(rho, 0)
                ax.plot(x, y, color=color[num])
        ax.set_ylabel(yname)
        ax.set_xlabel(xname)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)



            
if __name__ == "__main__":

    df = pd.read_csv('test.csv', index_col=0)
    df.fillna(0, inplace=True)
    df = df.loc[df['ViolateCausality']==False]

    drawer = EOSDrawer(df)
    ax = plt.subplot(111)
    drawer.DrawEOS(xlim=[1e-4,1e4], ylim=[1e-4, 1e4])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()

    """
    ax = DrawEOS(df, yname='GetSpeedOfSound', xlim=[1e-4,1e4], ylim=[1e-4, 1e4])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    """
            
