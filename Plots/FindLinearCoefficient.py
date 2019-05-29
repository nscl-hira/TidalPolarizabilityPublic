import scipy.integrate as integrate
import numpy as np
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import Plots.FillableHist as fhist
from Utilities.Utilities import ConcatenateListElements
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from Plots.DrawCorrelationMatrix import GetWeight, GetDeformabilityWeight, features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
  results = ['NoData', 'ViolateFrom', 'MaxMass', 'DensCentral(2)', 'ViolateCausality'] + ['lambda(%g)' % mass for mass in [1.2, 1.4, 1.6]]
  additional_info = ['P(1.5rho0)', 'P(2rho0)']
  
  orig_df = pd.DataFrame()
  features_to_be_shown = ['Lsym', 'Ksym', 'Ksat', 'Qsym', 'Qsat', 'Zsym', 'Zsat', 'msat']

  clf = []
  g = []
  mean_lambda = []
  for i in range(0, 5):
    with pd.HDFStore('Results/MetaNarrow5_%d.h5' % i) as store:
      print('Loading file %d' % i)

      new_df = pd.concat([ConcatenateListElements(store['kwargs'])[features], 
                          ConcatenateListElements(store['result'])[results], 
                          ConcatenateListElements(store['Additional_info'])[additional_info]], axis=1)
 
      chunk = new_df.copy()
      chunk = chunk[(chunk['lambda(1.4)'] > 0) & (chunk['lambda(1.4)'] < 1200)]
      chunk['weight'] = GetWeight(chunk)
      chunk['Postweight'] = GetDeformabilityWeight(chunk)

      def CausalityCut(df): 
        if 'ViolateFrom' in df:
          return ((df['ViolateFrom'] > df['DensCentral(2)']) | (df['ViolateCausality'] == False))
        else:
          return (df['ViolateCausality'] == False)
  
      chunk = chunk[CausalityCut(chunk)]
      if len(clf) == 0:
        for mass in [1.2, 1.4, 1.6]:
          mean_lambda.append(np.mean(chunk['lambda(%g)' % mass]))
          clf.append(Pipeline([('Stand', StandardScaler()), ('Reg', LinearRegression())]))
          clf[-1].fit(chunk[features_to_be_shown], chunk['lambda(%g)' % mass]/mean_lambda[-1], **{'Reg__sample_weight': chunk['Postweight']})
          print(clf[-1].named_steps['Reg'].coef_, features_to_be_shown)
      if len(g) == 0:
        for idx, mass in enumerate([1.2, 1.4, 1.6]):
          g.append(fhist.FillableHist2D(clf[idx].predict(chunk[features_to_be_shown])*mean_lambda[idx], chunk['lambda(%g)' % mass], bins=50, weights=chunk['Postweight']))
      else:
        for idx, mass in enumerate([1.2, 1.4, 1.6]):
          g[idx].Append(clf[idx].predict(chunk[features_to_be_shown])*mean_lambda[idx], chunk['lambda(%g)' % mass], weights=chunk['Postweight'])

  for graph in g:
    graph.Draw(plt.axes())
    plt.show()
