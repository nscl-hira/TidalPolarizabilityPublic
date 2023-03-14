import os
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.colors as colors
mpl.use('Agg')
from Utilities.EOSLoader import EOSLoader
from Plots.FillableHist import FillableHist2D
from Plots.DrawSym15 import target_name, target_mean, target_sd, GausProd
from Plots.DrawAcceptedEOSSpiRIT import GetMeanAndBounds, GetRanges
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import itertools as it
import logging
import scipy
from scipy.signal import savgol_filter
from scipy import interpolate
import pickle
from copy import copy
from mpi4py import MPI
from functools import partial


from Utilities.MasterSlave import MasterSlave

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['errorbar.capsize'] =  2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

logging.basicConfig(level=logging.CRITICAL)

def ContourToPatches(value, contour, **args):
    contour = [[x, y] for x, y in zip(value, contour)]
    path = pltPath.Path(contour)
    return path, patches.PathPatch(path, **args)


def GetHist(name, weighted, ranges):
    start = ranges[0]
    end = ranges[1]

    loader = EOSLoader(name, start=start, end=end)
    g = None
    result = loader.store.select('result', start=start, stop=end)
    density = loader.store.select('meta', start=0, stop=1)['rho'].iloc[0]
    pfracAll = loader.store.select('meta', start=start, stop=end)['pfrac']
    mufracAll = loader.store.select('meta', start=start, stop=end)['mufrac']

    if weighted:
        result.columns = [' '.join(col).strip() for col in result.columns.values]
        data = result.join(loader.store.select('Additional_info', start=start, stop=end))
        data = data.join(loader.store.select('kwargs', start=start, stop=end))
        weights = GausProd(np.atleast_2d(data[target_name].values), target_mean, target_sd)
    else:
        weights = np.array([1]*result.shape[0])

    first = False

    tot = 0#np.array([end - start]*density.shape[0])
    dUrca = np.array([0]*density.shape[0], dtype=np.float)
    if start == 0:
        first = True
    for i, weight in enumerate(weights):#range(num_load):
      if first:
         print(i, end='\r', flush=True)
      if loader.reasonable.iloc[i]:
        if np.isnan(weight) or weight == 0:
          continue
        pfrac = pfracAll.iloc[i]
        mufrac = mufracAll.iloc[i]
        efrac = pfrac - mufrac
        xe = efrac/pfrac

        if np.isnan(weight) or weight == 0:
          continue
        if g is None:
          g = FillableHist2D(density, pfrac, bins=[density.shape[0],1000], logy=False, range=[[density[0], density[density.shape[0]-1]], [0, 1]], smooth=False, weights=[weight]*density.shape[0])
        else:
          g.Append(density, pfrac, weights=[weight]*density.shape[0])
        dUrcaThreshold = 1/(1 + np.power(1 + np.power(xe, 1/3.), 3))
        idUrca = np.greater(pfrac, dUrcaThreshold)
        tot = tot + weight
        dUrca[idUrca] = dUrca[idUrca] + weight
        
    print(tot, dUrca)
    loader.Close()
    return g, np.array([tot]*density.shape[0]), dUrca

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
    print(' To use, enter\npython %s pdf_name input ....' % sys.argv[0])
  else:
    hist = None
    pdf_name = sys.argv[1]

    # draw example EOSs
    # draw boundary
    g = None
    ranges = GetRanges(sys.argv[2], mslave.size-1)
    GetHist(sys.argv[2], True, ranges[0])
    tot = None
    dUrca = None
    for gnew_Post, subTot, subdUrca in mslave.map(partial(GetHist, sys.argv[2], True), ranges, chunk_size=1):
        #for r in ranges:
        #gnew_Post, subTot, subdUrca = GetHist(sys.argv[2], True, r)
        if g is None:
            g = gnew_Post
            tot = subTot
            dUrca = subdUrca
        else:
            g += gnew_Post
            tot = tot + subTot
            dUrca = dUrca + subdUrca


    fig, ax = plt.subplots(figsize=(11,9))
    import matplotlib as mpl
    g.Draw(ax, norm=mpl.colors.LogNorm())#, cmap=mpl.cm.gray)

    plt.xlabel(r'$\rho$ (fm$^{-3}$)')
    plt.ylabel(r'Proton fraction')
    plt.xlim(0., 3*0.16)
    plt.tight_layout()
    plt.savefig(pdf_name)
    with open(pdf_name.replace('.pdf', '.pkl'), 'wb') as fid:
        pickle.dump(fig, fid)

    plt.clf()
    plt.plot(0.5*(g.xedge[1:] + g.xedge[:-1]), dUrca/tot)
    plt.xlabel(r'$\rho$ (fm$^{-3}$)')
    plt.ylabel('Direct Ucra prob.')
    plt.xlim(0., 3*0.16)
    plt.tight_layout()
    plt.savefig(pdf_name.replace('.pdf', '_ulcra.pdf'))

    

  mslave.Close()
