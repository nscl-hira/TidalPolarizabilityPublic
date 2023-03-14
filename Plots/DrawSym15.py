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

target_dens = [0.05,0.101,0.115,0.106,0.038,0.069,0.232]
features = ['Sym(%g)' % d for d in target_dens] + ['L(0.1)', 'L(0.232)', 'L(0.24)', 'P_Sym(2rho0)', 'P_Sym(2rho0)', 'Ksat', 'Mass1.44 R', 'Mass1.34 R', 'Mass1.4 Lambda', 'Mass1.8 R', 'Mass1.8 R']#, 'Mass2.07 R', 'Mass2.08 R']#
features_names = features

target_name = features
target_mean = [15.9, 24.7, 25.4, 25.5, 10.3, 16.8, 52, 71.5, 140.9, 151.25, 10.1, 10.3, 230, 13.02, 12.71, 190, 12.39, 13.7]
target_sd_low = [1, 0.8, 1.1, 1.1, 1.0, 1.2, 13, 22.6, 112.5, 105, 3, 2.8, 30, 1.06*math.sqrt(2), 1.19*math.sqrt(2), 120, 0.98*math.sqrt(2), 1.5*math.sqrt(2)]#, 76.75]#51*1.37]
target_sd_high = [1, 0.8, 1.1, 1.1, 1.0, 1.2, 13, 22.6, 112.5, 105, 3, 2.8, 30, 1.24*math.sqrt(2), 1.14*math.sqrt(2), 390, 1.3*math.sqrt(2), 2.6*math.sqrt(2)]#, 76.75]#51*1.37]

# reduce errorbar
#idList = []
#for id_, f in enumerate(features):
#    if f == 'Sym(0.232)' or f == 'L(0.232)' or f == 'L(0.24)':
#         #target_sd_low[id_] = 0.2*target_mean[id_]
#         #target_sd_high[id_] = 0.2*target_mean[id_]
#        idList.append(id_)
#
#for i in sorted(idList, reverse=True):
#    del target_name[i]
#    del target_mean[i]
#    del target_sd_low[i]
#    del target_sd_high[i]
  
target_mean = np.atleast_1d(target_mean)
target_sd = np.array([target_sd_low, target_sd_high]).T

def GausProd(x, mean, std):
    ans = np.exp(-0.5*np.power((x-mean)/std[:, 0], 2))
    ans[x > mean] = np.exp(-0.5*np.power((x-mean)/std[:, 1], 2))[x > mean]
    ans[np.isnan(ans)] = 0
    return np.prod(ans, axis=1)

def FormatNumWithErr(num, errUp, errLow, sigfig):
    """
    round number to sigfig, and truncate err accordingly
    if err is truncated to zero, we will take err's first sigfig
    return truncated num, truncted err and the decimal pt
    """
    if errUp < 0 or errLow < 0:
        raise Exception('Error cannot be negative')
    err = max(errUp, errLow)
    if num == 0 and err == 0:
        return 0, 0, sigfig

    sign = -1 if num < 0 else 1
    num = abs(num)
    # power of 10 of num
    nP10 = math.floor(math.log10(num)) + 1 if num != 0 else -math.inf
    # power of 10 below which truncation ocurs
    truncPt = nP10 - sigfig

    # power of 10 of err
    nP10err = math.floor(math.log10(err)) + 1 if err != 0 else -math.inf

    if nP10 == -math.inf:
        truncPt = nP10err - sigfig
    elif nP10err != -math.inf:
        truncPt = min(truncPt, nP10err-1)
    return sign*round(num, -truncPt), round(errUp, -truncPt), round(errLow, -truncPt), max(0, -truncPt)
    
    

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
    x_ranges = None
    y_ranges = None
    
    pdf_name = sys.argv[1]
    if(len(sys.argv) == 3):
      sys.argv.append(sys.argv[2])

    orig_df = pd.DataFrame()

    #all_obs = ['Sym(rho0)', 'L(0.67rho0)', 'L(rho0)', 'L(1.5rho0)', 'Mass1.4 R', 'Mass1.4 Lambda']#, 'Mass1.4 DensCentral']#    
    #all_names = [r'$S_0$', r'L(0.67$n_0$)', r'L($n_0$)', r'L(1.5$n_0$)', r'R(1.4M$_{\odot}$)', r'$\Lambda$(1.4M$_{\odot}$)']#, r'$n_c(1.4M_\odot)$']
    #all_units = ['MeV', 'MeV', 'MeV', 'MeV', 'km', ' ', 'fm$^{-3}$']
    #x_ranges = [[28, 42], [38, 75], [20, 140], [0, 350], [10.5, 15], [150, 1300]]
    #y_ranges = x_ranges

    all_obs = ['Ksat', 'Qsat', 'P_Sym(4rho0)', 'Esym', 'Lsym', 'Ksym', 'Qsym', 'Zsym']
    #all_obs = ['Ssym(0.1fm-3)', 'Lsym(0.1fm-3)', 'Ksym(0.1fm-3)', 'Qsym(0.1fm-3)', 'Zsym(0.1fm-3)']#'SINT0', 'SINT1', 'SINT2']#features + results
    all_names = all_obs
    all_units = ['']*len(all_obs)

    #all_obs = ['Sym(rho0)', 'L(rho0)', 'Mass1.4 R', 'Mass1.4 Lambda', 'P(1.5rho0)', 'P(2rho0)']
    #all_names = [r'$S(\rho_0)$', r'$L(\rho_0)$', r'R(1.4M$_{\odot}$)', r'$\Lambda$(1.4M$_{\odot}$)', r'$P_{PNM}(1.5\rho_0)$', r'$P_{PNM}(2\rho_0)$']
    #all_units = ['MeV', 'MeV', 'km', '', r'MeV/fm$^2$', r'MeV/fm$^2$']

    #all_obs = ['Sym(0.67rho0)', 'L(0.67rho0)', 'Ksym(0.67rho0)', 'Sym(rho0)', 'L(rho0)', 'Ksym', 'Mass1.4 R', 'Mass1.4 Lambda']#features + results
    #all_names = [r'$S(0.67\rho_0)$', r'$L(0.67\rho_0)$', r'$K_{sym}(0.67\rho_0)$', r'$S_0$', r'$L(\rho_0)$', r'$K_{sym}$', r'R(1.4M$_{\odot}$)', r'$\Lambda$(1.4M$_{\odot}$)']
    #all_units = ['MeV', 'MeV', 'MeV', 'MeV', 'MeV', 'MeV', 'km', '']

    with AnalyzeGenData(sys.argv[3]) as analyzer:
      for new_df, weight in analyzer.ReasonableData(all_obs):
        if prior is None:
          prior = [fhist.FillableHist(new_df[f].values, bins=50, color='red', normalize=True, smooth=True) for f in all_obs]
        else:
          for i, f in enumerate(all_obs):
            prior[i].Append(new_df[f].values)

    #  for new_df in analyzer.AllData(all_obs):
    #    if superPrior is None:
    #       superPrior = [fhist.FillableHist(new_df[f].values, bins=50, color='green', normalize=True, smooth=True) for f in all_obs]
    #    else:
    #       for i, f in enumerate(all_obs):
    #          superPrior[i].Append(new_df[f].values)

    #x_ranges = []
    #for i, obs in enumerate(all_obs):
    #    x_ranges.append(prior[i].range)
    #y_ranges = x_ranges

    with AnalyzeGenData(sys.argv[2]) as analyzer:
      for new_df, weight in analyzer.ReasonableData(list(set(features + results + all_obs)), 150000):
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

   
    g.Draw(fontsize=40, s=1, contour=True)
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

        #prior[i].Draw(g.axes2d[i, i], s=1)

        #xlim = g.axes2d[i, i].get_xlim()
        #if True: #all_obs[i] in features:
        #    superPrior[i].Draw(g.axes2d[i, i])
        #g.axes2d[i, i].set_xlim(*xlim)

        xmean = 0.5*(prior[i].edge[:-1] + prior[i].edge[1:])
        if False:#all_obs[i] == 'Ksat' or all_obs[i] == 'Mass1.4 Lambda':
            row.append('N.A.')
        else:
            try:
                cumsum = np.cumsum(prior[i].histogram)
                cumsum = cumsum/cumsum[-1]
                mean = xmean[np.argmax(cumsum >= 0.5)]
                low = xmean[np.argmax(cumsum >= 0.5 - 0.68/2)]
                up = xmean[np.argmax(cumsum >= 0.5 + 0.68/2)]
                pol = [1, mean, mean - low, up - mean]
                #pol,_ = curve_fit(asymGaus, xmean, prior[i].histogram, p0=[max(prior[i].histogram), np.average(xmean), 0.5*(xmean[-1] - xmean[0]),  0.5*(xmean[-1] - xmean[0])])
                cent, errLow, errUp, pt = FormatNumWithErr(pol[1], abs(pol[2]), abs(pol[3]), 3)
                pt = str(pt)
                row.append((r'$%.' + pt + r'f^{+%.'+ pt + r'f}_{-%.' + pt + r'f}$') % (cent, errUp, errLow))
            except Exception:
                row.append('N.A.')

        xmean = 0.5*(g.graphs[i][i].edge[:-1] + g.graphs[i][i].edge[1:])
        #def weighted_avg_and_std(values, weights):
        #    """
        #    Return the weighted average and standard deviation.
        #
        #    values, weights -- Numpy ndarrays with the same shape.
        #    """
        #    average = np.average(values, weights=weights)
        #    # Fast and numerically precise:
        #    variance = np.average((values-average)**2, weights=weights)
        #    ans = (average, math.sqrt(variance))
        #    print(ans)
        #    return ans
        #row.append(r'$%.1f\pm%.1f}$' % (np.average(xmean, weights=g.graphs[i][i].histogram), weighted_avg_and_std(xmean, g.graphs[i][i].histogram)[1]))
        try:
            cumsum = np.cumsum(g.graphs[i][i].histogram)
            cumsum = cumsum/cumsum[-1]
            mean = xmean[np.argmax(cumsum >= 0.5)]
            low = xmean[np.argmax(cumsum >= 0.5 - 0.68/2)]
            up = xmean[np.argmax(cumsum >= 0.5 + 0.68/2)]
            pol = [1, mean, mean - low, up - mean]
            cent, errLow, errUp, pt = FormatNumWithErr(pol[1], abs(pol[2]), abs(pol[3]), 3)
            pt = str(pt)
            row.append((r'$%.' + pt + r'f^{+%.'+ pt + r'f}_{-%.' + pt + r'f}$') % (cent, errUp, errLow))
            print('%f %f %f' % (cent, errUp, errLow))


            #pol,_ = curve_fit(asymGaus, xmean, g.graphs[i][i].histogram, p0=[max(g.graphs[i][i].histogram), np.average(xmean, weights=g.graphs[i][i].histogram), 0.5*(xmean[-1] - xmean[0]),  0.5*(xmean[-1] - xmean[0])])
            #if abs(pol[3]) > 1:
            #    row.append(r'$%.1f^{+%.1f}_{-%.1f}$' % (pol[1], abs(pol[3]), abs(pol[2])))
            #else:
            #    row.append(r'$%.2f^{+%.2f}_{-%.2f}$' % (pol[1], abs(pol[3]), abs(pol[2])))
        except Exception:
            row.append('N.A.')
        row.append(all_units[i])
        cell_text.append(row)
        
    gs = g.axes2d[0, 4].get_gridspec()
    axbig = g.fig.add_subplot(gs[0:3, 3:])
    table= axbig.table(cellText=cell_text, 
                       colLabels=['Prior', 'Posterior', ''],
                       colWidths=[0.35, 0.35, 0.3],
                       rowLabels=all_names,
                       loc='center right',
                       edges='open')
    #table.get_celld()[(0,0)].set_text_props(color='red')
    #table.get_celld()[(0,1)].set_text_props(color='blue')


    table.scale(0.7, 7)
    table.auto_set_font_size(False)
    table.set_fontsize(40)
    axbig.axis('off')
    axbig.axis('tight')

        
        #g.axes2d[i, i].text(0.2, 0.5, all_obs[i] + r' = $%.2f^{+%.2f}_{-%.2f}$' % (abs(pol[1]), abs(pol[3]), abs(pol[2])), transform=g.axes2d[i, i].transAxes, color='blue') 

    plt.savefig(pdf_name)
