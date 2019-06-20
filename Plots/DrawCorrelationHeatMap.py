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

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for correlation heatmap between variables')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input1 input2 ....' % sys.argv[0])
  else:
    g = {}
    
    results = ['lambda(%g)' % mass for mass in [1.2, 1.4, 1.6]]
    additional_info = ['P(1.5rho0)', 'P(2rho0)']
    
    orig_df = pd.DataFrame()
    features = ['Lsym', 'Ksym', 'P(2rho0)', 'Ksat', 'Qsym', 'Qsat']
    features_names = [r'$L_{sym}$', r'$K_{sym}$', r'$P(2\rho_0)$', r'$K_{sat}$', r'$Q_{sym}$', r'$Q_{sat}$']
  
    pdf_name = sys.argv[1]
    for filename in sys.argv[2:]:
      head, ext = os.path.splitext(filename)
      with pd.HDFStore(filename, 'r') as store, \
           pd.HDFStore(head + '.Weight' + ext, 'r') as weight_store:
  
        chunksize=80000
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

          if len(g) == 0:
            for mass in [1.2, 1.4, 1.6]:
              g['lambda(%g)' % mass] = []
              for feature in features:
                g['lambda(%g)' % mass].append(fhist.PearsonCorr(new_df[feature], 
                                              new_df['lambda(%g)' % mass], 
                                              post_weight, bins=100))
          else:
            for idx, feature in enumerate(features):
              for mass in [1.2, 1.4, 1.6]:
                g['lambda(%g)' % mass][idx].Append(new_df[feature], 
                                                   new_df['lambda(%g)' % mass], 
                                                   post_weight) 
   
    corr = []
    for mass in [1.2, 1.4, 1.6]:
      corr.append([])
      for idx, feature in enumerate(features):
        corr[-1].append(g['lambda(%g)' % mass][idx].corr_r)
  
    df = pd.DataFrame.from_dict(corr)
    df.columns = features
    df.index = ['lambda(%g)' % mass for mass in [1.2, 1.4, 1.6]]

    plt.figure(figsize=(12, 6)) 
    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9) 
    sns.heatmap(df, annot=True, 
                xticklabels=features_names, 
                yticklabels=[r'$\Lambda(%g)$' % mass for mass in [1.2, 1.4, 1.6]])
    plt.xticks(rotation='horizontal')
    plt.yticks(va='center')
    plt.tick_params(axis='both', which='both', length=0)
    plt.savefig(pdf_name)
