import matplotlib as mpl
mpl.use('Agg')
import scipy.integrate as integrate
import numpy as np
import os
import Plots.FillableHist as fhist
from Utilities.Utilities import ConcatenateListElements
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import sys

def CapAsymGaussian(x, mean, left_sd, right_sd):
    return np.piecewise(x, [x <=mean, x >mean], 
                        [lambda y: np.exp(-0.5*np.square((y - mean)/left_sd)), lambda y: np.exp(-0.5*np.square((y - mean)/(right_sd)))])

def NormalizedAsymGaussian(x, mean, left_sd, right_sd, a, b):
    area = integrate.quad(CapAsymGaussian, a, b, args=(mean, left_sd, right_sd))
    return CapAsymGaussian(x, mean, left_sd, right_sd)/area[0]

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for correlation matrix between variables')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input1 input2 ....' % sys.argv[0])
  else:
    g = None

    orig_df = pd.DataFrame()
    features = ['Lsym', 'Ksym', 'Ksat', 
                'Qsym', 'Qsat', 'Zsym', 
                'Zsat', 'msat', 'Mass1.4 Lambda', 'P(2rho0)']
    features_names = [r'$L_{sym}$', r'$K_{sym}$', r'$K_{sat}$', 
                      r'$Q_{sym}$', r'$Q_{sat}$', r'$Z_{sym}$', 
                      r'$Z_{sat}$', r'$m^{*}_{sat}$', r'$\Lambda(1.4)$', r'$P(2\rho_{0})$']

    pdf_name = sys.argv[1]
    for filename in sys.argv[2:]:
      head, ext = os.path.splitext(filename)
      with pd.HDFStore(filename, 'r') as store, \
           pd.HDFStore(head + '.Weight' + ext, 'r') as weight_store:

        new_mean = weight_store.get_storer('main').attrs.prior_mean
        new_sd = weight_store.get_storer('main').attrs.prior_sd

        chunksize = 8000
        for kwargs, result, add_info, weight in zip(store.select('kwargs', chunksize=chunksize), 
                                                    store.select('result', chunksize=chunksize), 
                                                    store.select('Additional_info', chunksize=chunksize),
                                                    weight_store.select('main', chunksize=chunksize)): 
          result.columns = [' '.join(col).strip() for col in result.columns.values]
          new_df = pd.concat([kwargs, result, add_info], axis=1)
          new_df = new_df[features]
          # only select reasonable data
          idx = (weight['Reasonable'].values & weight['Causality'].values).flatten()
          new_df = new_df[idx]
          weight = weight[idx]

          if g is None:
            g = fhist.FillablePairGrid(new_df, 
                                       weights=weight['PosteriorWeight'], 
                                       x_names=features_names, 
                                       y_names=features_names)
            g.map_lower(fhist.FillableHist2D, bins=100, cmap='inferno')
            g.map_upper(fhist.PearsonCorr, bins=100)
            g.map_diag(fhist.FillableHist, bins=50, normalize=True, color='r')
          else:
            g.Append(new_df, weights=weight['PosteriorWeight'])
    g.Draw()
    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.1, left=0.1, top=0.95)  
    g.fig.set_size_inches(25,25)
    g.fig.align_labels()#tight_layout()

    g.axes2d[2][0].set_ylim([249.167-2*26.833,249.167+2*26.833])
    g.axes2d[-1][2].set_xlim([249.167-2*26.833,249.167+2*26.833])
    for ax in g.axes2d[-2][:-2]:
      ax.set_ylim([250, 800])
      g.axes2d[-2][-2].set_xlim([250, 800])
    for ax in g.axes2d[-1][:-1]:
      ax.set_ylim([10, 50])
    g.axes2d[-1][-1].set_xlim([10, 50])

    # add prior to the plots
    for i, name in enumerate(features):
      try:
        xlim = g.axes2d[i, i].get_xlim()
        x = np.linspace(*xlim, 100)
        a, b = (xlim[0] - new_mean[name])/new_sd[name], (xlim[1] - new_mean[name])/new_sd[name]
        y = truncnorm.pdf(x, a, b, loc=new_mean[name], scale=new_sd[name])
        g.axes2d[i, i].plot(x, y, color='b')
      except:
        pass
    x = np.linspace(250, 800, 100)
    y = NormalizedAsymGaussian(x, 190, 120, 390, 250, 800)
    g.axes2d[-2, -2].plot(x, y, color='b')

    """
    print('name\tmean\tSD')
    for i, name in enumerate(features):
      print('%s\t%f\t%f' % (name, g.graphs[i][i].GetMean(), g.graphs[i][i].GetSD()))
    """

    plt.savefig(pdf_name)
