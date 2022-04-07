import matplotlib as mpl
mpl.use('Agg')
import os
import sys
import numpy as np
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import Plots.FillableHist as fhist
from Utilities.Utilities import ConcatenateListElements
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm
from scipy.optimize import curve_fit
import math

from AddWeight import AnalyzeGenData

all_masses = [1.4]
results = []#['Mass%g R' % mass for mass in all_masses] + ['Mass%g Lambda' % mass for mass in all_masses]# + ['Ksym', 'Ksat', 'Qsym', 'Qsat', 'Zsym', 'Zsat']
results_names = []#[r'R(%gM$_\odot$)' % m for m in all_masses] + [r'$\Lambda$(%gM$_\odot$)' % mass for mass in all_masses]

target_dens = [0.038, 0.05, 0.069, 0.101, 0.106, 0.115]
features = ['Sym(%g)' % d for d in target_dens] + ['L(0.1)', 'P_Sym(2rho0)', 'Sym(0.232)', 'Ksat', 'L(1.5rho0)', 'Mass1.4 R', 'Mass1.4 Lambda', 'Mass2.1 R']#
features_names = features

target_name = features
target_mean = [10.3, 15.9, 16.8, 24.7, 25.5, 25.4, 71.5, 10.245, 52, 230, 143.75, 13.02, 190, 12.39]#, 143.75]#61]
target_sd_low = [1, 1, 1.2, 0.8, 1.1, 1.1, 22.6, 2.90, 13, 30, 76.75, 1.06, 120, 0.98]#, 76.75]#51*1.37]
target_sd_high = [1, 1, 1.2, 0.8, 1.1, 1.1, 22.6, 2.90, 13, 30, 76.75, 1.24, 390, 1.3]#, 76.75]#51*1.37]


target_mean = np.atleast_1d(target_mean)
target_sd = np.array([target_sd_low, target_sd_high]).T

def GausProd(x, mean, std):
    ans = np.exp(-0.5*np.power((x-mean)/std[:, 0], 2))
    ans[x > mean] = np.exp(-0.5*np.power((x-mean)/std[:, 1], 2))[x > mean]
    ans[np.isnan(ans)] = 0
    return np.prod(ans, axis=1)

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for correlation heatmap between variables')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input_posterior input_prior' % sys.argv[0])
  else:
    g = None
    prior = None
    superPrior = None
    
    pdf_name = sys.argv[1]
    if(len(sys.argv) == 3):
      sys.argv.append(sys.argv[2])

    orig_df = pd.DataFrame()

    all_obs = ['Sym(rho0)', 'L(rho0)', 'L(1.5rho0)', 'Ksat', 'Mass1.4 R', 'Mass1.4 Lambda', 'Mass1.4 DensCentral']#    
    all_names = [r'$S_0$', r'L', r'L(1.5$\rho_0$)', 'K$_{sat}$', r'R(1.4M$_{\odot}$)', r'$\Lambda$(1.4M$_{\odot}$)', 'densc']
    all_units = ['MeV', 'MeV', 'MeV', 'Mev', 'km', ' ', ' ']
    #all_obs = ['Ksat', 'Qsat', 'Psym', 'SINT0', 'SINT1', 'SINT2']#features + results
    #all_names = all_obs
    #all_units = ['', '', '', '', '', '']


    with AnalyzeGenData(sys.argv[3]) as analyzer:
      for new_df, weight in analyzer.ReasonableData(all_obs):
        if prior is None:
          prior = [fhist.FillableHist(new_df[f].values, bins=50, color='red', normalize=True, smooth=True) for f in all_obs]
        else:
          for i, f in enumerate(all_obs):
            prior[i].Append(new_df[f].values)

      #for new_df in analyzer.AllData(all_obs):
      #  if superPrior is None:
      #     superPrior = [fhist.FillableHist(new_df[f].values, bins=50, color='green', normalize=True, smooth=True) for f in all_obs]
      #  else:
      #     for i, f in enumerate(all_obs):
      #        superPrior[i].Append(new_df[f].values)

    x_ranges = []
    for i, obs in enumerate(all_obs):
        x_ranges.append(prior[i].range)
    y_ranges = x_ranges

    with AnalyzeGenData(sys.argv[2]) as analyzer:
      for new_df, weight in analyzer.ReasonableData(list(set(features + results + all_obs))):
        if g is None:
          g = fhist.FillablePairGrid(new_df,
                                     weights=GausProd(new_df[features].values, target_mean, target_sd),#weight['PosteriorWeight']*np.exp(-np.power(new_df[features[0]] - target_mean, 2)/(2*target_sd*target_sd)), 
                                     x_vars=all_obs,
                                     x_names=all_names,
                                     y_vars=all_obs,
                                     y_names=all_names,
                                     x_ranges=x_ranges,
                                     y_ranges=y_ranges) 
          g.map_lower(fhist.FillableHist2D, bins=50, cmap='Blues', smooth=True)
          g.map_diag(fhist.FillableHist, bins=50, smooth=True, normalize=True, color='blue')
        else:
          g.Append(new_df, GausProd(new_df[features].values, target_mean, target_sd))#weight['PosteriorWeight']*np.exp(-np.power(new_df[features[0]] - target_mean, 2)/(2*target_sd*target_sd))))

   
    g.Draw(fontsize=40, s=1.5)
    plt.subplots_adjust(hspace=0.13, wspace=0.15, bottom=0.15, left=0.15, top=0.95)  
    g.fig.set_size_inches(35,25)
    #g.fig.set_size_inches(70,50)
    g.fig.align_labels()#tight_layout()

    def asymGaus(x, amp, mean, left_std, right_std):
        yl = amp*np.exp(-0.5*np.power((x-mean)/left_std, 2))
        yr = amp*np.exp(-0.5*np.power((x-mean)/right_std, 2))
        id = x < mean
        yr[id] = yl[id]
        return yr

    cell_text = []
    for i in range(len(all_obs)):
        row = []
        prior[i].Draw(g.axes2d[i, i], s=1.5)
        #xlim = g.axes2d[i, i].get_xlim()
        #if all_obs[i] in features:
        #  superPrior[i].Draw(g.axes2d[i, i])
        #g.axes2d[i, i].set_xlim(*xlim)

        xmean = 0.5*(prior[i].edge[:-1] + prior[i].edge[1:])
        if all_obs[i] == 'Ksat' or all_obs[i] == 'Mass1.4 Lambda':
            row.append('N.A.')
        else:
            try:
                pol,_ = curve_fit(asymGaus, xmean, prior[i].histogram, p0=[max(prior[i].histogram), np.average(xmean), 0.5*(xmean[-1] - xmean[0]),  0.5*(xmean[-1] - xmean[0])])
                row.append(r'$%.1f^{+%.1f}_{-%.1f}$' % (pol[1], abs(pol[3]), abs(pol[2])))

            except Exception:
                row.append('N.A.')

        xmean = 0.5*(g.graphs[i][i].edge[:-1] + g.graphs[i][i].edge[1:])
        try:
            pol,_ = curve_fit(asymGaus, xmean, g.graphs[i][i].histogram, p0=[max(g.graphs[i][i].histogram), np.average(xmean, weights=g.graphs[i][i].histogram), 0.5*(xmean[-1] - xmean[0]),  0.5*(xmean[-1] - xmean[0])])
            row.append(r'$%.1f^{+%.1f}_{-%.1f}$' % (pol[1], abs(pol[3]), abs(pol[2])))
        except Exception:
            row.append('N.A.')
        row.append(all_units[i])
        cell_text.append(row)
        
    gs = g.axes2d[0, 4].get_gridspec()
    axbig = g.fig.add_subplot(gs[0:3, 3:])
    table= axbig.table(cellText=cell_text, 
                       colLabels=['Before', 'After', ''],
                       colWidths=[0.4, 0.4, 0.2],
                       rowLabels=all_names,
                       loc='center right',
                       edges='open')
    table.get_celld()[(0,0)].set_text_props(color='red')
    table.get_celld()[(0,1)].set_text_props(color='blue')


    table.scale(0.7, 7)
    table.auto_set_font_size(False)
    table.set_fontsize(40)
    axbig.axis('off')
    axbig.axis('tight')

        
        #g.axes2d[i, i].text(0.2, 0.5, all_obs[i] + r' = $%.2f^{+%.2f}_{-%.2f}$' % (abs(pol[1]), abs(pol[3]), abs(pol[2])), transform=g.axes2d[i, i].transAxes, color='blue') 

    plt.savefig(pdf_name)
