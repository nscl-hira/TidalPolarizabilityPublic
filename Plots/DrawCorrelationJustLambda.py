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
    x_features = ['Lsym', 'Ksym', 'P(2rho0)', 'Ksat',
                'Qsym', 'Qsat']
    x_features_names = [r'$L_{sym}$', r'$K_{sym}$', r'$P(2\rho_0)$', r'$K_{sat}$', 
                        r'$Q_{sym}$', r'$Q_{sat}$']

    y_features = ['Mass1.2 Lambda', 'Mass1.4 Lambda', 'Mass1.6 Lambda']
    y_features_names = [r'$\Lambda(1.2)$', r'$\Lambda(1.4)$', r'$\Lambda(1.6)$']

    pdf_name = sys.argv[1]
    for filename in sys.argv[2:]:
      head, ext = os.path.splitext(filename)
      with pd.HDFStore(filename, 'r') as store, \
           pd.HDFStore(head + '.Weight' + ext, 'r') as weight_store:

        new_mean = weight_store.get_storer('main').attrs.prior_mean
        new_sd = weight_store.get_storer('main').attrs.prior_sd

        # set range to be within 2 sd
        x_bounds = []
        for name in x_features:
          if name in new_mean:
            x_bounds.append([new_mean[name] - 2*new_sd[name], new_mean[name] + 2*new_sd[name]])
          elif name == 'P(2rho0)':
            x_bounds.append([10, 50])

        chunksize = 8000
        for kwargs, result, add_info, weight in zip(store.select('kwargs', chunksize=chunksize), 
                                                    store.select('result', chunksize=chunksize), 
                                                    store.select('Additional_info', chunksize=chunksize),
                                                    weight_store.select('main', chunksize=chunksize)): 
          result.columns = [' '.join(col).strip() for col in result.columns.values]
          new_df = pd.concat([kwargs, result, add_info], axis=1)
          new_df = new_df[x_features + y_features]
          # only select reasonable data
          idx = (weight['Reasonable'].values & weight['Causality'].values).flatten()
          new_df = new_df[idx]
          weight = weight[idx]

          if g is None:
            g = fhist.FillablePairGrid(new_df, 
                                       weights=weight['PosteriorWeight'], 
                                       x_vars=x_features,
                                       x_names=x_features_names, 
                                       x_ranges=x_bounds,
                                       y_vars=y_features,
                                       y_names=y_features_names,
                                       y_ranges=[[1000, 2000], [250, 800], [100, 400]])
            g.map(fhist.FillableHist2D, bins=100, cmap='inferno')
          else:
            g.Append(new_df, weights=weight['PosteriorWeight'])
    g.Draw()
    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.2, left=0.1, top=0.95)  
    g.fig.set_size_inches(25,10)
    g.fig.align_labels()#tight_layout()

    print('name\tmean\tSD')
    for i, name in enumerate(y_features):
      print('%s\t%f\t%f' % (name, g.graphs[i][0].GetMean(1), g.graphs[i][0].GetSD(1)))

    plt.savefig(pdf_name)
