import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from prettytable import PrettyTable

from Plots.FillableHist import FillableHist2D
from Utilities.Utilities import ConcatenateListElements
from Plots.FindLinearCoefficient import all_masses


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

    all_mean_lambda = []
    all_std_lambda = []
    all_mean = []
    all_var = []
    all_coef = []
    all_intercepts = []

    x = PrettyTable()
    x.title = 'Parameter statistics'
    x.field_names = ['type'] + features
    var = coef_file[['%s_var' % feature for feature in features]].mean(axis=0).values
    mean = coef_file[['%s_mean' % feature for feature in features]].mean(axis=0).values
    x.add_row(['mean'] + ['%.2f' % val for val in mean])
    x.add_row(['Std'] + ['%.2f' % val for val in np.sqrt(var)])
    print(x)
    
    x = PrettyTable()
    x.field_names = ['Mass'] + features + ['Intercept', 'Mean lambda', 'Std lambda']
 
    for mass in all_masses:

      mass_coef = coef_file[coef_file['mass'] == mass]
      coef = mass_coef[['%s_coef' % feature for feature in features]].mean(axis=0).values
      intercept = mass_coef['intercept'].mean(axis=0)
      mean_lambda = mass_coef['mean_lambda'].mean(axis=0)
      std_lambda = mass_coef['std_lambda'].mean(axis=0)

      x.add_row(['%g' % mass] +['%.5f' % val for val in coef] + ['%.5f' % intercept, '%.5f' % mean_lambda, '%.5f' % std_lambda])

      all_mean_lambda.append(mean_lambda)
      all_std_lambda.append(std_lambda)
      all_coef.append(coef)
      all_mean.append(mean)
      all_var.append(var)
      all_intercepts.append(intercept)


    x.title = 'Coefficients'
    print(x)



    graphs = None#{1.2: None, 1.4: None, 1.6: None}
    graphs_no_std = None
    with pd.HDFStore(filename, 'r') as store, \
         pd.HDFStore(head + '.Weight' + ext, 'r') as weight_store:

      chunksize = 80000
      
      for kwargs, result, add_info, weight in zip(store.select('kwargs', chunksize=chunksize), 
                                                    store.select('result', chunksize=chunksize), 
                                                    store.select('Additional_info', chunksize=chunksize),
                                                    weight_store.select('main', chunksize=chunksize)): 
        result.columns = [' '.join(col).strip() for col in result.columns.values]
        new_df = pd.concat([kwargs, result, add_info], axis=1)
        # only select reasonable data
        idx = (weight['Reasonable'].values & weight['Causality'].values).flatten()
        new_df = new_df[idx]
        weight = weight[idx]

        for mean, var, coef, mass, intercept, mean_lambda, std_lambda in zip(all_mean, all_var, all_coef, all_masses, all_intercepts, all_mean_lambda, all_std_lambda):
          data = (new_df[features].values - mean)/np.sqrt(var)*coef 
          data = np.sum(data, axis=1) + intercept
          standarded_lambda = (new_df['Mass%g Lambda' % mass] - mean_lambda)/std_lambda
          intensity_factor = 1.
          """
          if mass == 1.2:
            intensity_factor = 5.
          elif mass == 1.4:
            intensity_factor = 2
          elif mass == 1.6:
            intensity_factor = 1.
          elif mass == 1.8:
            intensity_factor = 0.5
          elif mass == 2.:
            intensity_factor = 0.25
          """
          if graphs is None:
            graphs = FillableHist2D(standarded_lambda, data, intensity_factor*weight['PosteriorWeight'], bins=100, cmap='Reds', range=[[-2,2],[-2,2]])#, range=[[0,2000],[0,2000]])
          else:
            graphs.Append(standarded_lambda, data, intensity_factor*weight['PosteriorWeight'])

          if graphs_no_std is None and mass == 1.4:
            graphs_no_std = FillableHist2D(standarded_lambda*std_lambda + mean_lambda, new_df['Mass%g Lambda' % mass], intensity_factor*weight['PosteriorWeight'], bins=100, cmap='Reds', range=[[250,1000],[250,1000]])
          elif mass == 1.4:
            graphs_no_std.Append(standarded_lambda*std_lambda + mean_lambda, new_df['Mass%g Lambda' % mass], intensity_factor*weight['PosteriorWeight'])
          

    fig, ax = plt.subplots(2, figsize=(10,15))

    graphs.Draw(ax[0])
    ax[0].set_ylabel(r'$\hat{\Lambda}$ (TOV)')
    ax[0].set_xlabel(r'$\hat{\Lambda}$ (Linear model in Eq. (20))')
    line = np.linspace(-2, 2, 100)
    ax[0].plot(line, line, color='b')


    graphs_no_std.Draw(ax[1])
    ax[1].set_ylabel(r'$\Lambda$ (TOV)')
    ax[1].set_xlabel(r'$\Lambda$ (Linear model in Eq. (18))')
    line = np.linspace(0, 1000, 100)
    ax[1].plot(line, line, color='b')

    #lower = max([graph.xedge[0], graph.yedge[0]])
    #upper = min([graph.xedge[-1],graph.yedge[-1]])
    #plt.xlim([lower, upper])
    #plt.ylim([lower, upper])
    plt.tight_layout()
    plt.savefig('%s_linear_model.pdf' % pdf_name)
    #plt.clf()
        
        

