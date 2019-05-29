import scipy.integrate as integrate
import numpy as np
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import Plots.FillableHist as fhist
from Utilities.Utilities import ConcatenateListElements
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

new_mean = {'Esym': 32.775, 
            'Lsym': 69.86666667, 
            'Ksat':249.1666667, 
            'Ksym':-46.33333333, 
            'Qsat': -110.3333333, 
            'Qsym': 362.5, 
            'Zsat':3288.166667, 
            'Zsym':-3970.833333, 
            'msat': 0.731666667, 
            'kv':0.41}
new_sd = {'Esym': 1.875, 
          'Lsym': 22.51666667,
          'Ksat': 26.83333333, 
          'Ksym': 82.33333333, 
          'Qsat': 233.8333333, 
          'Qsym': 252.5, 
          'Zsat': 1159.333333, 
          'Zsym': 1754.5, 
          'msat': 0.081666667, 
          'kv': 0.216666667}

def GetWeight(df, new_mean=new_mean, new_sd=new_sd):
    order = ['Esym','Lsym','Ksat','Ksym','Qsat','Qsym','Zsat','Zsym','msat','kv']
    data = np.array([df[name] for name in order]).T
    ordered_mean = np.array([new_mean[name] for name in order]).reshape(1,-1)
    ordered_sd = np.array([new_sd[name] for name in order]).reshape(1,-1)

    deg_freedom = len(order)
    exp = np.exp(-0.5*np.sum(np.square((data - ordered_mean)/ordered_sd), axis=1))
    #pre_factor = np.power(2*np.pi, deg_freedom/2.)*np.prod(ordered_sd)
    return exp#/pre_factor

def GetDeformabilityWeight(df):
    exp = ((df['lambda(1.4)'] > 190)*np.exp(-0.5*np.square((df['lambda(1.4)'] - 190)/390)) + (df['lambda(1.4)'] <= 190)*np.exp(-0.5*np.square((df['lambda(1.4)'] - 190)/120)))*df['weight']
    return exp

features = ['Esym', 'Lsym', 'Ksym', 'Qsym', 'Ksat', 'Qsat', 'Zsat', 'Zsym', 'msat', 'kv']

def CapAsymGaussian(x, mean, left_sd, right_sd):
    return np.piecewise(x, [x <=mean, x >mean], 
                        [lambda y: np.exp(-0.5*np.square((y - mean)/left_sd)), lambda y: np.exp(-0.5*np.square((y - mean)/(right_sd)))])

def NormalizedAsymGaussian(x, mean, left_sd, right_sd, a, b):
    area = integrate.quad(CapAsymGaussian, a, b, args=(mean, left_sd, right_sd))
    return CapAsymGaussian(x, mean, left_sd, right_sd)/area[0]

if __name__ == '__main__':
  g = None

  
  results = ['NoData', 'ViolateFrom', 'MaxMass', 'DensCentral(2)', 'ViolateCausality', 'lambda(1.4)']
  additional_info = ['P(1.5rho0)', 'P(2rho0)']
  
  orig_df = pd.DataFrame()
  features_to_be_shown = ['Lsym', 'Ksym', 'Ksat', 'Qsym', 'Qsat', 'Zsym', 'Zsat', 'msat', 'lambda(1.4)']
  features_names = [r'$L_{sym}$', r'$K_{sym}$', r'$K_{sat}$', r'$Q_{sym}$', r'$Q_{sat}$', r'$Z_{sym}$', r'$Z_{sat}$', r'$m^{*}_{sat}$', r'$\Lambda(1.4)$']

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

      if g is None:
        g = fhist.FillablePairGrid(chunk[features_to_be_shown], weights=chunk['Postweight'], x_names=features_names, y_names=features_names)
        g.map_lower(fhist.FillableHist2D, bins=100, cmap='inferno')
        g.map_upper(fhist.PearsonCorr, bins=100)
        g.map_diag(fhist.FillableHist, bins=50, normalize=True, color='r')
      else:
        g.Append(chunk[features_to_be_shown], weights=chunk['Postweight'])
  g.Draw()
  plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.1, left=0.1, top=0.95)  
  g.fig.set_size_inches(25,25)
  g.fig.align_labels()#tight_layout()
  for ax in g.axes2d[-1][:-1]:
    ax.set_ylim([250, 800])
  g.axes2d[-1][-1].set_xlim([250, 800])


  # add prior to the plots
  for i, name in enumerate(features_to_be_shown):
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
  g.axes2d[-1, -1].plot(x, y, color='b')

  print('name\tmean\tSD')
  for i, name in enumerate(features_to_be_shown):
    print('%s\t%f\t%f' % (name, g.graphs[i][i].GetMean(), g.graphs[i][i].GetSD()))

  plt.savefig('Correlation.pdf')
