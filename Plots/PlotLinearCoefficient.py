import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from Plots.FillableHist import FillableHist2D
from Utilities.Utilities import ConcatenateListElements


if __name__ == '__main__':
  if len(sys.argv) <= 3:
    print('This script is used to plot the goodness of the linear fit')
    print('python %s Coefficient_csv data pdf_names' % sys.argv[0])
  else:
    coef_file = sys.argv[1]
    filename = sys.argv[2]
    pdf_name = sys.argv[3]
    head, ext = os.path.splitext(filename)
    coef_file = pd.read_csv(coef_file, sep=',')
    features = [var.replace('_mean', '') for var in list(coef_file) if var.endswith('_mean')]
    lambda_list = ['lambda(%g)' % mass for mass in [1.2, 1.4, 1.6]]

    all_mean = []
    all_var = []
    all_coef = []
    all_mass = [1.2, 1.4, 1.6]
    all_intercepts = []
    for mass in all_mass:
      mass_coef = coef_file[coef_file['mass'] == mass]
      var = mass_coef.filter(regex='.*_var').mean(axis=0).values
      mean = mass_coef.filter(regex='.*_mean').mean(axis=0).values
      coef = mass_coef.filter(regex='.*_coef').mean(axis=0).values
      intercept = mass_coef['intercept'].mean(axis=0)
      print('mass %g' % mass)
      print('\t' + '\t'.join(features))
      print('mean\t%s' % ('\t'.join(['%.2f' % val for val in mean])))
      print('Std\t%s' % ('\t'.join(['%.2f' % np.sqrt(val) for val in var])))
      print('Coef\t%s' % ('\t'.join(['%.2f' % val for val in coef])))
      print('Intercept\t%.2f' % intercept)
      all_coef.append(coef)
      all_mean.append(mean)
      all_var.append(var)
      all_intercepts.append(intercept)
      
 

    graphs = {1.2: None, 1.4: None, 1.6: None}
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
   
        new_df = new_df[features + lambda_list]
        # only select reasonable data
        idx = reasonable & causality
        new_df = new_df[idx]
        prior_weight = prior_weight[idx]
        post_weight = post_weight[idx]

        for mean, var, coef, mass, intercept in zip(all_mean, all_var, all_coef, all_mass, all_intercepts):
          data = (new_df[features].values - mean)/np.sqrt(var)*coef 
          data = np.sum(data, axis=1) + intercept
          if graphs[mass] is None:
            graphs[mass] = FillableHist2D(data, new_df['lambda(%g)' % mass], post_weight, bins=50, cmap='inferno')
          else:
            graphs[mass].Append(data, new_df['lambda(%g)' % mass], post_weight)

    for mass, graph in graphs.items():
      graph.Draw(plt.axes())
      plt.xlabel(r'$\Lambda(%g)$' % mass)
      plt.ylabel('Linear model')
      lower = max([graph.xedge[0], graph.yedge[0]])
      upper = min([graph.xedge[-1],graph.yedge[-1]])
      plt.xlim([lower, upper])
      plt.ylim([lower, upper])
      line = np.linspace(lower, upper, 1000)
      plt.plot(line, line, color='b')
      plt.savefig('%s_mass_%g.pdf' % (pdf_name, mass))
      plt.clf()
        
        

