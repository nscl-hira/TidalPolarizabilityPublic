import sys
from pathos.multiprocessing import ProcessingPool
from pathos.helpers import cpu_count
import scipy.optimize as opt
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


from functools import partial
from tqdm import tqdm

import Utilities.SkyrmeEOS as sky
from Utilities.Constants import *
from Utilities.EOSCreator import EOSCreator


def GetEOS(name_and_row):
    name = name_and_row[0]
    row = name_and_row[1]
    #eos = sky.Skryme(row)
    creator = EOSCreator(row)
    kwargs = creator.PrepareEOS(**row) 
    eos, trans_dens = creator.GetEOSType(**kwargs)
    #eos.ToFile('AllSkyrmes/EOS_%s.txt' % name)
    #print('finished %s' % name)
    return name, eos, trans_dens, creator.rho, creator.pfrac, creator.mufrac

class EOSDrawer:


    def __init__(self, df, ncpu=20, **kwargs):
        self.df = df
        name_list = [(index, row) for index, row in df.iterrows()]
        pool = ProcessingPool(nodes=ncpu)
        result = pool.imap(GetEOS, name_list)
        self.EOS = {}
        print('Preparing EOS in progress:')
        for val in tqdm(result, total=len(name_list), unit='EOS', ncols=100, **kwargs):
            self.EOS[val[0]] = val[1::]
        #pool.close()
        #pool.join()
        pool.terminate()
        pool.restart()
        

    def ParticleFraction(self, name):
        return tuple(self.EOS[name][2::])

    def Merge(self, other_drawers):
        for drawer in other_drawers:
            self.EOS.update(drawer.EOS)

    def Hist2DEOS(self, df=None, xlim=None, ylim=None, bins=50, logx=False, logy=False):
        rho0 = 0.16
        histogram = np.zeros((bins, bins), dtype=int)

        if logx:
            xbins = np.logspace(*np.log(xlim), num=bins+1, base=np.e)
        else:
            xbins = np.linspace(*xlim, num=bins+1)

        if logy:
            ybins = np.logspace(*np.log(ylim), num=bins+1, base=np.e)
        else:
            ybins = np.linspace(*ylim, num=bins+1)

        for index, row in df.iterrows():
            eos, trans_dens, _, _, _ = self.EOS[index]
            trans_dens = [10*rho0] + trans_dens + [1e-9]
            for num, (low_den, high_den) in enumerate(zip(trans_dens, trans_dens[1:])):
                rho = np.linspace(low_den, high_den, 100)
                y = eos.GetPressure(rho, 0)
                newhist, xedges, yedges = np.histogram2d(rho, y, bins=[xbins, ybins])
                histogram = histogram + newhist
        return histogram, xedges, yedges
        

    def DrawEOS(self, df=None, ax=None, xname='GetEnergyDensity', yname='GetPressure', xlim=None, ylim=None, color=['r', 'b', 'g', 'orange', 'b', 'pink'], labels=[], **kwargs):

        # dict containing lines and its name
        index_list = {}

        if ax is None:
            ax = plt.subplot(111)
        rho = np.concatenate([np.logspace(np.log(1e-9), np.log(3.76e-4), 10, base=np.exp(1)), np.linspace(3.77e-4, 1.6, 90)])
        if df is None:
            df = self.df
        first = True
        for index, row in df.iterrows():
            index_list[index] = []
            if 'rho0' in row:
                rho0 = row['rho0']
            else:
                rho0 = 0.16

            eos, trans_dens, _, _, _ = self.EOS[index]
            trans_dens = [10*rho0] + trans_dens + [1e-9]
            for num, (low_den, high_den) in enumerate(zip(trans_dens, trans_dens[1:])):
                rho = np.linspace(low_den, high_den, 1000)
                if xname == 'rho':
                    x = rho
                elif xname == 'rho/rho0':
                    x = rho/rho0
                else:
                    x = getattr(eos, xname)(rho, 0)
                if yname == 'rho':
                    y = rho
                else:
                    y = getattr(eos, yname)(rho, 0)
                label=None
                if len(labels) > 0 and first:
                    label = labels[num]
                line, = ax.plot(x, y, color=color[num], label=label, **kwargs)
                index_list[index].append(line)
            first = False

        #ax.set_ylabel(yname)
        #ax.set_xlabel(xname)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        return index_list


"""        
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

    ax = DrawEOS(df, yname='GetSpeedOfSound', xlim=[1e-4,1e4], ylim=[1e-4, 1e4])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
"""
            
