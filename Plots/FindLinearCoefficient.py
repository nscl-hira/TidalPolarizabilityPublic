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

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for correlation matrix between variables')
    print('Input: List of filenames from deformability calculation')
    print('Output: csv files of the fitted coefficient')
    print(' To use, enter\npython %s csv_name input1 input2 ....' % sys.argv[0])
  else:
   
    results = ['lambda(%g)' % mass for mass in [1.2, 1.4, 1.6]]
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

        chunksize = 80000
        for kwargs, result, add_info, \
            reasonable, causality, prior_weight, \
            post_weight in zip(store.select('kwargs', chunksize=chunksize),
                               store.select('result', chunksize=chunksize),
                               store.select('Additional_info', chunksize=chunksize),
                               weight_store.select('Reasonable', chunksize=chunksize),
                               weight_store.select('Causality', chunksize=chunksize),
                               weight_store.select('PriorWeight', chunksize=chunksize),
                               weight_store.select('PosteriorWeight', chunksize=chunksize)): 
 
          new_df = pd.concat([ConcatenateListElements(kwargs), 
                              ConcatenateListElements(result), 
                              ConcatenateListElements(add_info)], axis=1)
   
          new_df = new_df[features + results]
          # only select reasonable data
          idx = reasonable & causality
          new_df = new_df[idx]
          prior_weight = prior_weight[idx]
          post_weight = post_weight[idx]


          for mass in [1.2, 1.4, 1.6]:
            info = {}
            info['mean_lambda'] = np.mean(new_df['lambda(%g)' % mass])
            clf = Pipeline([('Stand', StandardScaler()), ('Reg', LinearRegression())])
            clf.fit(new_df[features], 
                    new_df['lambda(%g)' % mass], 
                    **{'Reg__sample_weight': post_weight})
            for feature, mean in zip(features, clf.named_steps['Stand'].mean_):
              info['%s_mean' % feature] = mean
            for feature, var in zip(features, clf.named_steps['Stand'].var_):
              info['%s_var' % feature] = var
            for feature, coef in zip(features, clf.named_steps['Reg'].coef_):
              info['%s_coef' % feature] = coef
            info['intercept'] = clf.named_steps['Reg'].intercept_
            info['mass'] = mass
            all_info.append(info)
    pd.DataFrame(all_info).to_csv(csv_name)

