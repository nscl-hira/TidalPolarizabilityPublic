import numpy as np
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import Plots.FillableHist as fhist
from Utilities.Utilities import ConcatenateListElements
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm
from Plots.DrawCorrelationMatrix import GetWeight, GetDeformabilityWeight, features

if __name__ == '__main__':
  g = {}
  
  results = ['NoData', 'ViolateFrom', 'MaxMass', 'DensCentral(2)', 'ViolateCausality']
  for mass in [1.2, 1.4, 1.6]:
    results += ['lambda(%g)' % mass]
  additional_info = ['P(1.5rho0)', 'P(2rho0)']
  
  orig_df = pd.DataFrame()
  features_to_be_shown = ['Lsym', 'Ksym', 'P(2rho0)', 'Ksat', 'Qsym', 'Qsat']
  features_names = [r'$L_{sym}$', r'$K_{sym}$', r'$P(2\rho_0)$', r'$K_{sat}$', r'$Q_{sym}$', r'$Q_{sat}$']

  for i in range(0, 18):
    with pd.HDFStore('Results/MetaNarrow5_%d.h5' % i) as store:

      new_df = pd.concat([ConcatenateListElements(store['kwargs'])[features], 
                          ConcatenateListElements(store['result'])[results], 
                          ConcatenateListElements(store['Additional_info'])[additional_info]], axis=1)
 
      chunk = new_df.copy()#[features + ['lambda(1.4)',]]
      chunk = chunk[(chunk['lambda(1.4)'] > 0) & (chunk['lambda(1.4)'] < 1200)]
      chunk['weight'] = GetWeight(chunk)
      chunk['Postweight'] = GetDeformabilityWeight(chunk)

      def CausalityCut(df): 
        if 'ViolateFrom' in df:
          return ((df['ViolateFrom'] > df['DensCentral(2)']) | (df['ViolateCausality'] == False))
        else:
          return (df['ViolateCausality'] == False)
  
      chunk = chunk[CausalityCut(chunk)]
      if len(g) == 0:
        for mass in [1.2, 1.4, 1.6]:
          g['lambda(%g)' % mass] = []
          for feature in features_to_be_shown:
            g['lambda(%g)' % mass].append(fhist.PearsonCorr(chunk[feature], chunk['lambda(%g)' % mass], chunk['Postweight'], bins=100))
      else:
        for idx, feature in enumerate(features_to_be_shown):
          for mass in [1.2, 1.4, 1.6]:
            g['lambda(%g)' % mass][idx].Append(chunk[feature], chunk['lambda(%g)' % mass], chunk['Postweight']) 
 
  corr = []
  for mass in [1.2, 1.4, 1.6]:
    corr.append([])
    for idx, feature in enumerate(features_to_be_shown):
      corr[-1].append(g['lambda(%g)' % mass][idx].corr_r)


  df = pd.DataFrame.from_dict(corr)
  df.columns = features_to_be_shown
  df.index = ['lambda(%g)' % mass for mass in [1.2, 1.4, 1.6]]
  sns.heatmap(df, annot=True, xticklabels=features_names, yticklabels=[r'$\Lambda(%g)$' % mass for mass in [1.2, 1.4, 1.6]])
  plt.xticks(rotation='horizontal')
  plt.yticks(va='center')
  plt.tick_params(axis='both', which='both', length=0)
  plt.show()
