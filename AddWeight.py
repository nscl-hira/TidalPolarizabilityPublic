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
    exp = (df['lambda(1.4)'] > 190)*np.exp(-0.5*np.square((df['lambda(1.4)'] - 190)/390)) \
          + (df['lambda(1.4)'] <= 190)*np.exp(-0.5*np.square((df['lambda(1.4)'] - 190)/120))
    return exp

def CausalityCut(df): 
  if 'ViolateFrom' in df:
    return ((df['ViolateFrom'] > df['DensCentral(2)']) | (df['ViolateCausality'] == False))
  else:
    return (df['ViolateCausality'] == False)
    

features = ['Esym', 'Lsym', 'Ksym', 'Qsym', 'Ksat', 'Qsat', 'Zsat', 'Zsym', 'msat', 'kv']
results = ['NoData', 'ViolateFrom', 'MaxMass', 'DensCentral(2)', 'ViolateCausality', 'lambda(1.4)', 'lambda(1.2)', 'lambda(1.6)']

def Summarize(args):
  new_df = pd.concat([ConcatenateListElements(args[0])[features], 
                      ConcatenateListElements(args[1])[results]], axis=1, sort=False)
  reasonable = (new_df['lambda(1.4)'] > 0) & (new_df['lambda(1.4)'] < 1200) & (new_df['lambda(1.2)'] > 0) & (new_df['lambda(1.6)'] > 0)
  prior_weight = GetWeight(new_df)
  posterior_weight = GetDeformabilityWeight(new_df)*prior_weight
  causality = CausalityCut(new_df)

  return reasonable, pd.DataFrame(prior_weight, index=new_df.index), posterior_weight, causality

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
        for arg in zip(store.select('kwargs', chunksize=chunksize), store.select('result', chunksize=chunksize)):
          result = Summarize(arg)
          weight.append('Reasonable', result[0], min_itemsize=10)
          weight.append('PriorWeight', result[1], min_itemsize=10)
          weight.append('PosteriorWeight', result[2], min_itemsize=10)
          weight.append('Causality', result[3], min_itemsize=10)
        weight.get_storer('PriorWeight').attrs.prior_mean = new_mean
        weight.get_storer('PriorWeight').attrs.prior_sd = new_sd

