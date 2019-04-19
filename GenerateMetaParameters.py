import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('EOSComparsion.csv')
    pars = ['Esat', 'Esym', 'Lsym', 'Ksat', 'Ksym', 'Qsat', 'Qsym', 'Zsat', 'Zsym', 'msat', 'kv']
    priors = []
    for model in set(df['Name']):
        Average = df[(df['Name'] == model) & (df['Type'] == 'Average')][pars].iloc[0]
        Sigma = df[(df['Name'] == model) & (df['Type'] == 'Sigma')][pars].iloc[0]

        Esat, Esym, Lsym, Ksat, Ksym, Qsat, Qsym, Zsat, Zsym, msat, kv = np.random.uniform(Average - 2*Sigma, Average + 2*Sigma, size=(100000, Average.shape[0])).T

        new_prior = pd.DataFrame({'Esat':Esat.flatten(), 
                                  'Esym':Esym.flatten(), 
                                  'Lsym':Lsym.flatten(), 
                                  'Ksat':Ksat.flatten(), 
                                  'Ksym':Ksym.flatten(), 
                                  'Qsat':Qsat.flatten(), 
                                  'Qsym':Qsym.flatten(), 
                                  'Zsat':Zsat.flatten(), 
                                  'Zsym':Zsym.flatten(), 
                                  'msat':msat.flatten(), 
                                  'kv':kv.flatten()})
        new_prior['Model_Type'] = model
        priors.append(new_prior)

    priors = pd.concat(priors)
    priors.to_csv('test.csv')
