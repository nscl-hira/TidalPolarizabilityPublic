import sys
import os
import pandas as pd

foldername = 'ABrownUDen'

data = {'Model':[], 't0':[], 't1':[], 't2':[], 't31':[], 't32':[], 't33':[], 
        'x0':[], 'x1':[], 'x2':[], 'x31':[], 'x32':[], 'x33':[],
        'sigma1':[], 'sigma2':[], 'sigma3':[], 'rho0':[]}
for filename in os.listdir(foldername):
  if filename.endswith('.den'):
    with open(os.path.join(foldername, filename)) as file_:
      values = file_.readlines()
      line = values[4].split(',')
      data['Model'].append(filename.rstrip('.den'))
      data['t0'].append(line[0])
      data['t1'].append(line[1])
      data['t2'].append(line[2])
      data['t31'].append(line[3])
      data['t32'].append(0)
      data['t33'].append(0)
      line = values[5].split(',')
      data['x0'].append(line[0])
      data['x1'].append(line[1])
      data['x2'].append(line[2])
      data['x31'].append(line[3])
      data['x32'].append(0)
      data['x33'].append(0)
      data['sigma1'].append(line[4])
      data['sigma2'].append(0)
      data['sigma3'].append(0)
      data['rho0'] = 0.16
      
df = pd.DataFrame.from_dict(data)
df = df.set_index('Model')
print(df)
df.to_csv('%s.csv'%foldername)

