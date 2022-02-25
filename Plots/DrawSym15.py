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

from AddWeight import AnalyzeGenData

all_masses = [1.4]
results = ['Mass%g R' % mass for mass in all_masses] + ['Mass%g Lambda' % mass for mass in all_masses]# + ['Ksym', 'Ksat', 'Qsym', 'Qsat', 'Zsym', 'Zsat']
results_names = [r'R(%gM$_\odot$)' % m for m in all_masses] + [r'$\Lambda$(%gM$_\odot$)' % mass for mass in all_masses]

features = ['Sym(0.67rho0)', 'L(0.67rho0)', 'Sym(1.5rho0)', 'L(1.5rho0)', 'P_Sym(4rho0)']
features_names = [r'$S(0.67\rho_0)$', r'$L(0.67\rho_0)$', r'$S(1.5\rho_0)$', r'$L(1.5\rho_0)$', r'$P_{SM}(4\rho_0)$']

target_name = features
target_mean = [25, 71.46, 46, 61, 127.39]#91.77]
target_sd = [1, 22.6, 8, 51, 72.31]#8]#50.8]#60.6]

target_mean = np.atleast_1d(target_mean)
target_sd = np.atleast_1d(target_sd)

def GausProd(x, mean, std):
    ans = np.exp(-0.5*np.power((x-mean)/std, 2))
    return np.prod(ans, axis=1)

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for correlation heatmap between variables')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input1 input2 ....' % sys.argv[0])
  else:
    g = None
    prior = None
    superPrior = None
    
    pdf_name = sys.argv[1]
    orig_df = pd.DataFrame()

    all_obs = features + results
    all_names = features_names + results_names

    for filename in sys.argv[2:]:
      with AnalyzeGenData(filename) as analyzer:
        for new_df, weight in analyzer.ReasonableData(features + results):
          if g is None:
            g = fhist.FillablePairGrid(new_df,
                                       weights=GausProd(new_df[features].values, target_mean, target_sd),#weight['PosteriorWeight']*np.exp(-np.power(new_df[features[0]] - target_mean, 2)/(2*target_sd*target_sd)), 
                                       x_vars=all_obs,
                                       x_names=all_names,
                                       y_vars=all_obs,
                                       y_names=all_names) 
            g.map_lower(fhist.FillableHist2D, bins=100, cmap='Blues')
            g.map_diag(fhist.FillableHist, bins=100, smooth=True, normalize=True, color='blue')
            #g.map_upper(fhist.PearsonCorr, bins=100)

            prior = [fhist.FillableHist(new_df[f].values, bins=100, color='red', normalize=True) for f in all_obs]
          else:
            g.Append(new_df, GausProd(new_df[features].values, target_mean, target_sd))#weight['PosteriorWeight']*np.exp(-np.power(new_df[features[0]] - target_mean, 2)/(2*target_sd*target_sd))))
            for i, f in enumerate(all_obs):
                prior[i].Append(new_df[f].values)

    for filename in sys.argv[2:]:
      with AnalyzeGenData(filename) as analyzer:
        for new_df in analyzer.AllData(features + results):
          if superPrior is None:
             superPrior = [fhist.FillableHist(new_df[f].values, bins=100, color='green', normalize=True) for f in all_obs]
          else:
             for i, f in enumerate(all_obs):
                superPrior[i].Append(new_df[f].values)

   
    g.Draw(fontsize=40)
    plt.subplots_adjust(hspace=0.13, wspace=0.15, bottom=0.15, left=0.15, top=0.95)  
    g.fig.set_size_inches(35,25)
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
        prior[i].Draw(g.axes2d[i, i])
        if all_obs[i] in features:
            superPrior[i].Draw(g.axes2d[i, i])

        xmean = 0.5*(prior[i].edge[:-1] + prior[i].edge[1:])
        pol,_ = curve_fit(asymGaus, xmean, prior[i].histogram, p0=[max(prior[i].histogram), np.average(xmean), 0.5*(xmean[-1] - xmean[0]),  0.5*(xmean[-1] - xmean[0])])
        row.append(r'$%.1f^{+%.1f}_{-%.1f}$' % (pol[1], abs(pol[3]), abs(pol[2])))

        xmean = 0.5*(g.graphs[i][i].edge[:-1] + g.graphs[i][i].edge[1:])
        pol,_ = curve_fit(asymGaus, xmean, g.graphs[i][i].histogram, p0=[max(g.graphs[i][i].histogram), np.average(xmean), 0.5*(xmean[-1] - xmean[0]),  0.5*(xmean[-1] - xmean[0])])
        row.append(r'$%.1f^{+%.1f}_{-%.1f}$' % (pol[1], abs(pol[3]), abs(pol[2])))
        cell_text.append(row)
        
    gs = g.axes2d[0, 4].get_gridspec()
    axbig = g.fig.add_subplot(gs[0:4, 4:])
    table= axbig.table(cellText=cell_text, 
                       colLabels=['Before', 'After'],
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
