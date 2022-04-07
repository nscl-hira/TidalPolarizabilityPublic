import os
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.colors as colors
mpl.use('Agg')
from Utilities.EOSLoader import EOSLoader
from Plots.FillableHist import FillableHist2D
from Plots.DrawSym15 import target_name, target_mean, target_sd, GausProd
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools as it
import logging
import scipy
from scipy.signal import savgol_filter
from scipy import interpolate
import pickle
import copy
from mpi4py import MPI
from functools import partial


from Utilities.MasterSlave import MasterSlave


logging.basicConfig(level=logging.CRITICAL)

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def GetMeanAndBounds(g, CI=0.68):
    hist = copy.deepcopy(g.histogram.T)
    id = np.sum(g.histogram.T, axis=0) > 0
    hist = hist[:, id]
    y_percentile = np.cumsum(hist, axis=0)/np.sum(hist, axis=0)
    y = 0.5*(g.yedge[1:] + g.yedge[:-1])
    x = 0.5*(g.xedge[1:] + g.xedge[:-1])
    x = x[id]

    idUpper = y_percentile > (1+CI)/2
    #findUpper = copy.deepcopy(y_percentile)
    #findUpper[idUpper] = 0

    idLower = y_percentile > (1-CI)/2
    #findLower = copy.deepcopy(y_percentile)
    #findLower[idLower] = 1

    idMean = y_percentile > 0.5
    #findMean = copy.deepcopy(y_percentile)
    #findMean[idMean] = 1

    upperBound = y[np.argmax(idUpper, axis=0)]
    mean = y[np.argmax(idMean, axis=0)]
    lowerBound = y[np.argmax(idLower, axis=0)]

    
    id = (lowerBound == 0) 
    lowerBound[id] = mean[id]
    id = upperBound == 0
    upperBound[id] = mean[id]
    #id = np.sum(upperBound) == 0
    #upperBound[id] = lowerBound[id]

    
    #return x, mean, lowerBound, upperBound
    return x, savgol_filter(mean, 25, 3), savgol_filter(lowerBound, 25, 3), savgol_filter(upperBound, 25, 3)

def GetRanges(filename, ncores, num_load=None):
    ranges = []
    if num_load is None:
        with pd.HDFStore(sys.argv[2], 'r') as store:
            num_load = store.get_storer('kwargs').nrows
    batch_size = int(num_load/ncores)
    start = 0
    for i in range(ncores):
        ranges.append([start, start + batch_size if i != ncores - 1 else num_load - 1])
        start = start + batch_size
    return ranges    



def GetHist(name, weighted, ranges):
    start = ranges[0]
    end = ranges[1]

    loader = EOSLoader(name, start=start, end=end)
    g = None 
    g_Post = None
    result = loader.store.select('result', start=start, stop=end)
    if weighted:
        result.columns = [' '.join(col).strip() for col in result.columns.values]
        data = result.join(loader.store.select('Additional_info', start=start, stop=end))
        data = data.join(loader.store.select('kwargs', start=start, stop=end))
        weights = GausProd(np.atleast_2d(data[target_name].values), target_mean, target_sd)
    else:
        weights = np.array([1]*(result.shape[0]))

    first = False
    if start == 0:
        first = True
    for i, weight in enumerate(weights):#range(num_load):
      if first:
         print(i, end='\r', flush=True)
      if loader.reasonable.iloc[i]:
        if np.isnan(weight) or weight == 0:
          continue

        eos, density = loader.GetNSEOS(i)
        density = np.linspace(1e-2, 5*0.16, 100)
        pressure = eos.GetPressure(density)
        energy = eos.GetEnergyDensity(density) 
        
        if g_Post is None:
          #g = FillableHist2D(density, pressure, bins=[100,10000], logy=False, range=[[1e-2, 5*0.16], [-100, 1e3]], smooth=False, weights=[1]*density.shape[0])
          g_Post = FillableHist2D(density, pressure, bins=[100,10000], logy=False, range=[[1e-2, 5*0.16], [-100, 1e3]], smooth=False, weights=[weight]*density.shape[0])
        else:
          #g.Append(density, pressure, weights=[1]*density.shape[0])
          g_Post.Append(density, pressure, weights=[weight]*density.shape[0])
    loader.Close()
    print('finish')
    return g_Post

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
logging.basicConfig(filename='log/app_rank%d.log' % rank, format='Process id %(process)d: %(name)s %(levelname)s - %(message)s', level=logging.DEBUG)
#logging.basicConfig(format='Process id %(process)d: %(name)s %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
  mslave = MasterSlave(comm)
  if len(sys.argv) <= 2:
    print('This script generates pdf images for rejected EOS')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input_posterior input_prior ....' % sys.argv[0])
  else:
    with pd.HDFStore(sys.argv[2], 'r') as store:
        num_load = store.get_storer('kwargs').nrows
    CI = 0.95

    # first draw Ksym vs Lsym
    hist = None
    pdf_name = sys.argv[1]
    if(len(sys.argv) == 3):
      sys.argv.append(sys.argv[2])


    # draw example EOSs
    # draw boundary
    g = None
    g_Post = None
    ranges = GetRanges(sys.argv[3], mslave.size-1)
    print(ranges)
    for gnew in mslave.map(partial(GetHist, sys.argv[3], False), ranges, chunk_size=1):
        if g is None:
            g = gnew
        else:
            g += gnew

    ranges = GetRanges(sys.argv[2], mslave.size-1)
    print(ranges)
    for gnew_Post in mslave.map(partial(GetHist, sys.argv[2], True), ranges, chunk_size=1):
        if g_Post is None:
            g_Post = gnew_Post
        else:
            g_Post += gnew_Post


    fig, ax = plt.subplots(figsize=(14,10))
    #import matplotlib as mpl
    #g.Draw(ax, norm=mpl.colors.LogNorm())#, cmap=mpl.cm.gray)
    x, mean, lowerB, upperB = GetMeanAndBounds(g, CI=CI)
    ax.fill_between(x, lowerB, upperB, alpha=1, edgecolor='blue', facecolor='none', linestyle='--', label='%g%% C.I. without constraints' % (CI*100))

    x, mean, lowerB, upperB = GetMeanAndBounds(g_Post, CI=CI)#_Post)
    #plt.plot(x, mean, label='mean', color='green')
    ax.fill_between(x, lowerB, upperB, alpha=1, color='aqua', label='%g%% C.I. with constraints' % (CI*100))

    CI = 0.68
    x, mean, lowerB, upperB = GetMeanAndBounds(g_Post, CI=CI)
    ax.fill_between(x, lowerB, upperB, alpha=1, color='green', label='%g%% C.I. with constraints' % (CI*100))


    plt.yscale('log')
    plt.xlabel(r'$\rho$ fm$^{-3}$')
    plt.ylabel(r'P (MeV/fm$^{3}$)')
    plt.ylim(1e-1, 1e3)
    plt.xlim(0, 5*0.16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name)
    with open(pdf_name.replace('.pdf', '.pkl'), 'wb') as fid:
        pickle.dump(fig, fid)

  mslave.Close()
