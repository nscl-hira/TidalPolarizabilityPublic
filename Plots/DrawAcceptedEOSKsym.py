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
from Utilities.Utilities import DataIO

logging.basicConfig(level=logging.CRITICAL)

def GetRes(name, ranges):
    start = ranges[0]
    end = ranges[1]

    loader = EOSLoader(name, start=start, end=end)
    first = False
    if start == 0:
        first = True
    data = []
    for i in range(len(loader)):
      #i = i + start
      if first:
         print(i, end='\r', flush=True)
      eos = loader.GetNuclearEOS(i)
      ele = {'Ssym(0.1fm-3)' : eos.GetAsymEnergy(0.1), 
             'Lsym(0.1fm-3)' : eos.GetL(0.1), 
             'Ksym(0.1fm-3)' : eos.GetKsym(0.1), 
             'Qsym(0.1fm-3)' : eos.GetQsym(0.1), 
             'Zsym(0.1fm-3)' : eos.GetZsym(0.1)}
      data.append([loader.GetName(i), ele])
    loader.Close()
    return data

# modifications
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
logging.basicConfig(filename='log/app_rank%d.log' % rank, format='Process id %(process)d: %(name)s %(levelname)s - %(message)s', level=logging.DEBUG)
#logging.basicConfig(format='Process id %(process)d: %(name)s %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
  mslave = MasterSlave(comm)
  if len(sys.argv) < 2:
    print('This script generates HDF5 files for Ksym (the original HDF5 does not store Ksym info).')
    print('Input: List of filenames from deformability calculation')
    print('Output: HDF5 with additional parameters')
    print(' To use, enter\npython %s input' % sys.argv[0])
  else:
    CI = 0.95

    # first draw Ksym vs Lsym
    hist = None
    filename = sys.argv[1]
    head, ext = os.path.splitext(filename)

    dataIO = DataIO(head + '.ExtInfo' + ext, flush_interval=1000)
    ranges = GetRanges(filename, mslave.size-1)
    print(ranges)
    for dataList in mslave.ordered_map(partial(GetRes, filename), ranges, chunk_size=1):
        for data in dataList:
            dataIO.AppendData('ExtInfo', data[0], data[1])
    dataIO.Close()

  mslave.Close()
