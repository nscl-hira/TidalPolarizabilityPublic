import matplotlib as mpl
mpl.use('Agg')
import scipy.integrate as integrate
import numpy as np
import os
import Plots.FillableHist as fhist
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import sys
from scipy.optimize import curve_fit
import math

from AddWeight import AnalyzeGenData

def PowerLaw(inv_comp, a, b):
  return a*np.power(inv_comp, b)

# helper function in printing scientific notation
def frexp10(x):
    exp = int(math.floor(math.log10(abs(x))))
    return x / 10**exp, exp

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for correlation matrix between variables')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input1 input2 ....' % sys.argv[0])
  else:
    g = None

    orig_df = pd.DataFrame()
    all_masses = [1.2, 1.4, 1.6, 1.8]
    features = ['Mass%g Lambda' % mass for mass in all_masses] + ['Mass%g R' % mass for mass in all_masses]

    pdf_name = sys.argv[1]
    for filename in sys.argv[2:]:
      with AnalyzeGenData(filename) as analyzer:
        for new_df, weight in analyzer.ReasonableData(features): 
          for mass in all_masses:
            lambda_ = new_df['Mass%g Lambda' % mass]
            inv_comp = new_df['Mass%g R' % mass]/mass
            keep = ~np.isnan(lambda_)
            if g is None:
              g = fhist.FillableHist2D(inv_comp[keep], lambda_[keep], weight['PosteriorWeight'][keep], cmap='Reds', range=[[6, 10], [0,1000]], bins=100)
            else:
              g.Append(inv_comp[keep], lambda_[keep], weight['PosteriorWeight'][keep])
    g.Draw(plt.axes())

    """
    curve fitting
    """
    data = g.histogram.ravel()
    yy, xx = np.meshgrid(0.5*(g.ybins[:-1] + g.ybins[1:]), 0.5*(g.xbins[:-1] + g.xbins[1:]))
    keep = data > 0
    data = data[keep]
    xx = xx.ravel()[keep]
    yy = yy.ravel()[keep]
    noise = 1/np.sqrt(data)
    popt, pcov = curve_fit(PowerLaw, xx, yy, sigma=noise)
    x = np.linspace(g.xbins[0], g.xbins[-1], 100)
    plt.plot(x, PowerLaw(x, *popt), color='b', label=r'$\Lambda=$%.2f$\times$10$^{%d}$(R/M)$^{%.2f}$' % (*frexp10(popt[0]), popt[1]))
    plt.xlabel('R/M (km/solar mass)')
    plt.ylabel(r'$\Lambda$')
    plt.legend()
    plt.savefig(pdf_name)
