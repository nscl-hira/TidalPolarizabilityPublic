import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.colors as colors
#mpl.use('Agg')
from Utilities.EOSLoader import EOSLoader
from Plots.FillableHist import FillableHist2D
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

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for rejected EOS')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input1 input2 ....' % sys.argv[0])
  else:
    num_load = 5000

    # first draw Ksym vs Lsym
    hist = None
    pdf_name = sys.argv[1]

    loader = EOSLoader(sys.argv[2])
    accepted_params = []
    rejected_params = []
    # draw example EOSs
    # draw boundary
    g = None 
    E_sym_2rho0 = []
    E_sym_weight = []
    for i in range(num_load):
      #if not loader.NoNSEOS(i):
      if loader.reasonable.iloc[i]:
        eos, density = loader.GetNSEOS(i)
        density = np.logspace(np.log(1e-7), np.log(5*0.16), 500, base=np.e)
        pressure = eos.GetPressure(density)
        energy = eos.GetEnergyDensity(density) 
        weight = loader.weight.iloc[i]
 
        if np.isnan(weight) or weight == 0:
          continue
        if g is None:
          g = FillableHist2D(energy, pressure, bins=100, logx=True, logy=True, range=[[1e1, 5e2], [1e-2, 3e2]], smooth=False, weights=[weight]*energy.shape[0])
        else:
          g.Append(energy, pressure, weights=[weight]*energy.shape[0])
        eos = loader.GetNuclearEOS(i)
        E_sym_2rho0.append(eos.GetAsymEnergy(2*0.16))
        E_sym_weight.append(weight)
    E_sym_2rho0 = np.array(E_sym_2rho0)
    E_sym_weight = np.array(E_sym_weight)
    print('Mean Esym(2rho0) : %f' % np.average(E_sym_2rho0))
    print('Esym(2rho0) STD : %f' % weighted_avg_and_std(E_sym_2rho0, E_sym_weight)[1])

    fig, ax = plt.subplots(figsize=(7,5))
    y_repeated = np.tile(0.5*(g.yedge[1:] + g.yedge[:-1]).reshape(-1,1), (1, g.histogram.T.shape[1]))
    mean = np.average(y_repeated, axis=0, weights=g.histogram.T)
 
    upper_idx = (y_repeated - mean.reshape(1,-1)) > 0
    lower_idx = (y_repeated - mean.reshape(1,-1)) < 0
    weight_lower = g.histogram.T.copy()
    weight_lower[upper_idx] = 0
    weight_upper = g.histogram.T.copy()
    weight_upper[lower_idx] = 0
    upper_variance = np.average((y_repeated - mean.reshape(1,-1))**2, weights=weight_upper, axis=0)
    lower_variance = np.average((y_repeated - mean.reshape(1,-1))**2, weights=weight_lower, axis=0)

    #interval = scipy.stats.norm.interval(0.95, loc=mean, scale=np.sqrt(variance))
    x = 0.5*(g.xedge[1:] + g.xedge[:-1])
    plt.plot(x, np.exp(savgol_filter(np.log(mean), 51, 3)), label='mean')
    ax.fill_between(x, np.exp(savgol_filter(np.log(mean - 2*np.sqrt(lower_variance)), 51, 3)), np.exp(savgol_filter(np.log(mean + 2*np.sqrt(upper_variance)), 51, 3)), alpha=0.3, color='r', label=r'2$\sigma$ region')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\mathcal{E}$ (MeV/fm$^{3}$)')
    plt.ylabel('P (MeV/fm$^{3}$)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name)
    loader.Close()
