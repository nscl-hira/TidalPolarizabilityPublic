import matplotlib as mpl
mpl.use('Agg')
import os
import sys
import numpy as np
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import Plots.FillableHist as fhist
from Utilities.Utilities import ConcatenateListElements
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm

from AddWeight import AnalyzeGenData

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for correlation heatmap between variables')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input1 input2 ....' % sys.argv[0])
  else:
    g = {}
    
    all_masses = [1, 1.2, 1.4, 1.6, 1.8, 2]
    results = ['Mass%g Lambda' % mass for mass in all_masses]
    
    orig_df = pd.DataFrame()
    features = ['Lsym', 'Ksym', 'P(2rho0)', 
                'Ksat', 'Qsym', 'Qsat']
    features_names = [r'$L_{sym}$ (MeV)', r'$K_{sym}$ (MeV)', r'$P(2\rho_0)$ (MeV/fm$^{3})$', 
                      r'$K_{sat}$ (MeV)', r'$Q_{sym}$ (MeV)', r'$Q_{sat}$ (MeV)']
  
    pdf_name = sys.argv[1]
    for filename in sys.argv[2:]:
      with AnalyzeGenData(filename) as analyzer:
        for new_df, weight in analyzer.ReasonableData(features + results):
          if len(g) == 0:
            for mass in all_masses:
              g['lambda(%g)' % mass] = []
              for feature in features:
                g['lambda(%g)' % mass].append(fhist.PearsonCorr(new_df[feature], 
                                              new_df[('Mass%g Lambda' % mass)], 
                                              weight['PosteriorWeight'], bins=100))
          else:
            for idx, feature in enumerate(features):
              for mass in all_masses:
                g['lambda(%g)' % mass][idx].Append(new_df[feature], 
                                                   new_df[('Mass%g Lambda' % mass)], 
                                                   weight['PosteriorWeight']) 
   
    corr = []
    for mass in all_masses:
      corr.append([])
      for idx, feature in enumerate(features):
        corr[-1].append(g['lambda(%g)' % mass][idx].corr_r)
  
    df = pd.DataFrame.from_dict(corr)
    df.columns = features
    df.index = ['lambda(%g)' % mass for mass in all_masses]

    plt.figure(figsize=(12, 6)) 
    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9) 
    sns.heatmap(df, annot=True, 
                xticklabels=features_names, 
                yticklabels=[r'$\Lambda(%g)$' % mass for mass in all_masses], cmap='Reds')
    plt.xticks(ha='right', rotation=20, fontsize=20)
    plt.yticks(va='center', fontsize=20)
    plt.tick_params(axis='both', which='both', length=0)
    plt.tight_layout()
    plt.savefig(pdf_name)
