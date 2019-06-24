import numpy as np
import sys
import os
from Utilities.Utilities import ConcatenateListElements
import pandas as pd

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
"""
new_sd = {'Esym': 1.875, 
          'Lsym': 22.51666667,
          'Ksat': 50, 
          'Ksym': 82.33333333, 
          'Qsat': 233.8333333, 
          'Qsym': 252.5, 
          'Zsat': 1159.333333, 
          'Zsym': 1754.5, 
          'msat': 0.081666667, 
          'kv': 0.216666667}
"""

def GetWeight(df, new_mean=new_mean, new_sd=new_sd):
    order = ['Esym','Lsym','Ksat','Ksym','Qsat','Qsym','Zsat','Zsym','msat','kv']
    data = np.array([df[name] for name in order]).T
    ordered_mean = np.array([new_mean[name] for name in order]).reshape(1,-1)
    ordered_sd = np.array([new_sd[name] for name in order]).reshape(1,-1)

    deg_freedom = len(order)
    exp = np.exp(-0.5*np.sum(np.square((data - ordered_mean)/ordered_sd), axis=1))
    return exp

def GetDeformabilityWeight(df):
    lambda_1_4 = df[('Mass1.4', 'Lambda')].values
    exp = (lambda_1_4 > 190)*np.exp(-0.5*np.square((lambda_1_4 - 190)/390)) \
          + (lambda_1_4 <= 190)*np.exp(-0.5*np.square((lambda_1_4 - 190)/120))
    return exp

def CausalityCut(df): 
  if 'ViolateFrom' in df:
    return ((df['ViolateFrom'] > df[('Mass2', 'DensCentral')]) | (df['ViolateCausality'] == False))
  else:
    return (df['ViolateCausality'] == False)
    

features = ['Esym', 'Lsym', 'Ksym', 
            'Qsym', 'Ksat', 'Qsat', 
            'Zsat', 'Zsym', 'msat', 
            'kv']
results = [('Mass1.2', 'Lambda'), 
           ('Mass1.4', 'Lambda'), 
           ('Mass1.6', 'Lambda'), 
           ('MaxMass', 'DensCentral'), 
           ('MaxMass', 'Mass'),
           ('Mass2', 'DensCentral')]
check_eos = ['NoData', 'ViolateFrom', 'ViolateCausality']

def Summarize(args):
  new_df = pd.concat([args[0][features], args[1][results], args[2][check_eos]], axis=1, sort=False)
  reasonable = (new_df[('Mass1.4', 'Lambda')] > 0) & (new_df[('Mass1.4', 'Lambda')] < 1200) & (new_df[('Mass1.2', 'Lambda')] > 0) & (new_df[('Mass1.6', 'Lambda')] > 0) & (new_df[('MaxMass', 'Mass')] > 2)
  prior_weight = pd.Series(GetWeight(new_df), index=new_df.index)
  posterior_weight = GetDeformabilityWeight(new_df)*prior_weight
  causality = CausalityCut(new_df)

  result = pd.concat([reasonable, 
                      prior_weight, \
                      posterior_weight, \
                      causality], axis=1) 
  result.columns = ['Reasonable', 'PriorWeight', 'PosteriorWeight', 'Causality']
  result.index = new_df.index
  return result

if __name__ == '__main__':
  if len(sys.argv) == 1:
    print('This script adds weight to the calculation result.')
    print('Input: List of filenames from deformability calculation')
    print('Output: HDF5 with weights of each EOS. Output name = path_to_data/Weight_{Input filename}')
    print(' To use, enter\npython %s file1 file2 ....' % sys.argv[0])
  else:
    for filename in sys.argv[1:]:
      head, ext = os.path.splitext(filename)
      with pd.HDFStore(filename, 'r') as store, \
           pd.HDFStore(head + '.Weight' + ext, 'w') as weight:
        chunksize=50000
        for arg in zip(store.select('kwargs', chunksize=chunksize), 
                       store.select('result', chunksize=chunksize),
                       store.select('EOSCheck', chunksize=chunksize)):
          result = Summarize(arg)
          weight.append('main', result, min_itemsize=30)
        weight.get_storer('main').attrs.prior_mean = new_mean
        weight.get_storer('main').attrs.prior_sd = new_sd

