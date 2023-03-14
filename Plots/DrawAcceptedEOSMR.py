import os
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.colors as colors
#mpl.use('Agg')
from Utilities.EOSLoader import EOSLoader
from Plots.FillableHist import FillableHist2D
from Plots.DrawSym15 import target_name, target_mean, target_sd, GausProd
from Plots.DrawAcceptedEOSSpiRIT import GetMeanAndBounds, GetRanges
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from AddWeight import AnalyzeGenData
import itertools as it
import logging
import scipy
from scipy.signal import savgol_filter
from scipy import interpolate
import pickle
from mpi4py import MPI 
from functools import partial 
import copy

from Utilities.MasterSlave import MasterSlave 

logging.basicConfig(level=logging.CRITICAL)

def GetHist(name, weighted, ranges):
    start = ranges[0]
    end = ranges[1]

    loader = EOSLoader(name, start=start, end=end)
    result = loader.store.select('result', start=start, stop=end)
    if weighted:
        result.columns = [' '.join(col).strip() for col in result.columns.values]
        data = result.join(loader.store.select('Additional_info', start=start, stop=end))
        data = data.join(loader.store.select('kwargs', start=start, stop=end))
        weights = GausProd(np.atleast_2d(data[target_name].values), target_mean, target_sd)
    else:
        weights = np.array([1]*(result.shape[0]))

    #M = np.array([x/10. for x in range(10, 21, 2)] + [2.17])
    M = np.array([0.3,0.4,0.45,0.5,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.07,2.08,2.17])
    Rs = []
    for m in M:
      Rs.append(loader.store.select('result', start=start, stop=end)['Mass%g' % m]['R'].values)

    Rs = np.array(Rs).T

    g_Post = None

    first = False
    if start == 0:
        first = True

    for i, (R, weight) in enumerate(zip(Rs, weights)):#range(num_load):
        if first:
           print(i, end='\r', flush=True)
        if not loader.reasonable.iloc[i] or weight == 0 or np.isnan(weight):
           continue
        tck, u = interpolate.splprep([M, R], s=0)
        unew, out = interpolate.splev(np.linspace(0, 1.01, 400), tck)
 
        if g_Post is None:
          g_Post = FillableHist2D(unew, out, bins=[100, 500], range=[[0.3, 2.17], [5, 25]], smooth=False, weights=[weight]*out.shape[0])
        else:
          g_Post.Append(unew, out, weights=[weight]*out.shape[0])
    loader.Close()
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

    num_load = 1500
    CI = 0.95

    # first draw Ksym vs Lsym
    hist = None
    pdf_name = sys.argv[1]

    # draw example EOSs
    # draw boundary
    g = None
    g_Post = None
    ranges = []

    ranges = GetRanges(sys.argv[3], mslave.size-1)
    print(ranges)
    #for r in ranges: 
    #    gnew = GetHist(sys.argv[3], False, r)
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
    plt.savefig(pdf_name)
    x, mean, lowerB, upperB = GetMeanAndBounds(g, CI=CI)
    ax.fill_betweenx(x, lowerB, upperB, alpha=1, edgecolor='blue', facecolor='none', linestyle='--', label='%g%% C.I. prior' % (CI*100), zorder=3)

    x, mean, lowerB, upperB = GetMeanAndBounds(g_Post, CI=CI)#_Post)
    ax.fill_betweenx(x, lowerB, upperB, alpha=1, color='aqua', label='%g%% C.I. posterior' % (CI*100), zorder=0)
    CI = 0.68
    x, mean, lowerB, upperB = GetMeanAndBounds(g_Post, CI=CI)#_Post)
    ax.fill_betweenx(x, lowerB, upperB, alpha=1, color='green', label='%g%% C.I. posterior' % (CI*100), zorder=1)


    plt.xlabel(r'$R$ (km)')
    plt.ylim(0.3, 2.17)
    plt.ylabel(r'$M$ ($\odot$)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name)
    with open(pdf_name.replace('.pdf', '.pkl'), 'wb') as fid:
        pickle.dump(fig, fid)
  mslave.Close()
