import sys
import os
import scipy.integrate as integrate
import numpy as np
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import Plots.FillableHist as fhist
from Utilities.Utilities import ConcatenateListElements
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

all_masses = [1.2, 1.4, 1.6, 1.8, 2.]

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for correlation matrix between variables')
    print('Input: List of filenames from deformability calculation')
    print('Output: csv files of the fitted coefficient')
    print(' To use, enter\npython %s csv_name input1 input2 ....' % sys.argv[0])
  else:
   

    results = ['Mass%g Lambda' % mass for mass in all_masses]
    additional_info = ['P(1.5rho0)', 'P(2rho0)']
    
    orig_df = pd.DataFrame()
    features = ['Lsym', 'Ksym', 'Ksat', 'Qsym', 'Qsat', 'Zsym', 'Zsat', 'msat']
  
    clf = []
    g = []
    csv_name = sys.argv[1]
    all_info = []
    for filename in sys.argv[2:]:
      head, ext = os.path.splitext(filename)
      with pd.HDFStore(filename, 'r') as store, \
           pd.HDFStore(head + '.Weight' + ext, 'r') as weight_store:

        chunksize = 100000
        for kwargs, result, add_info, weight in zip(store.select('kwargs', chunksize=chunksize), 
                                                    store.select('result', chunksize=chunksize), 
                                                    store.select('Additional_info', chunksize=chunksize),
                                                    weight_store.select('main', chunksize=chunksize)): 
          result.columns = [' '.join(col).strip() for col in result.columns.values]
          new_df = pd.concat([kwargs, result, add_info], axis=1)
          new_df = new_df[features + results]
          # only select reasonable data
          idx = (weight['Reasonable'].values & weight['Causality'].values).flatten()
          new_df = new_df[idx]
          weight = weight[idx]


          for mass in all_masses:
            info = {}
            info['mean_lambda'] = np.average(new_df['Mass%g Lambda' % mass], weights=weight['PosteriorWeight'], axis=0)
            info['std_lambda'] = np.sqrt(np.average((new_df['Mass%g Lambda' % mass] - info['mean_lambda'])**2, weights=weight['PosteriorWeight'], axis=0))
            standardized_lambda = (new_df['Mass%g Lambda' % mass] - info['mean_lambda'])/info['std_lambda']
            
            stand_mean = np.average(new_df[features], weights=weight['PosteriorWeight'].values.flatten(), axis=0)
            stand_std = np.sqrt(np.average((new_df[features] - stand_mean)**2, weights=weight['PosteriorWeight'], axis=0))
            stand_features = (new_df[features] - stand_mean)/stand_std

            clf = LinearRegression()
            clf.fit(stand_features, 
                    standardized_lambda, 
                    sample_weight=weight['PosteriorWeight'])
              
            for feature, mean in zip(features, stand_mean):
              info['%s_mean' % feature] = mean
            for feature, var in zip(features, np.square(stand_std)):
              info['%s_var' % feature] = var
            for feature, coef in zip(features, clf.coef_):
              info['%s_coef' % feature] = coef
            info['intercept'] = clf.intercept_
            info['mass'] = mass
            all_info.append(info)
    pd.DataFrame(all_info).to_csv(csv_name)

