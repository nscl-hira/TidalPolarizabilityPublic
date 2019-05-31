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
    all_coef = pd.read_csv(coef_file, sep=',')
    features = [var.replace('_mean', '') for var in list(all_coef) if var.endswith('_mean')]
    lambda_list = ['lambda(%g)' % mass for mass in [1.2, 1.4, 1.6]]


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

        for mass in [1.2, 1.4, 1.6]:
          mass_coef = all_coef[all_coef['mass'] == mass]
          var = mass_coef.filter(regex='.*_var').mean(axis=0).values
          mean = mass_coef.filter(regex='.*_mean').mean(axis=0).values
          coef = mass_coef.filter(regex='.*_coef').mean(axis=0).values
          data = (new_df[features].values - mean)/np.sqrt(var)*coef
          data = np.sum(data, axis=1)
          if graphs[mass] is None:
            graphs[mass] = FillableHist2D(data, new_df['lambda(%g)' % mass], post_weight, bins=50)
          else:
            graphs[mass].Append(data, new_df['lambda(%g)' % mass], post_weight)

    for mass, graph in graphs.items():
      graph.Draw(plt.axes())
      plt.savefig('%s_mass_%g.pdf' % (pdf_name, mass))
      plt.clf()
        
        

