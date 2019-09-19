import matplotlib as mpl
mpl.use('Agg')
import scipy.integrate as integrate
import numpy as np
import os
import Plots.FillableHist as fhist
from Utilities.Utilities import ConcatenateListElements
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import sys

from AddWeight import AnalyzeGenData

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for correlation matrix between variables')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input1 input2 ....' % sys.argv[0])
  else:
    g = None

    orig_df = pd.DataFrame()
    features = ['Lsym', 'Ksym', 'Ksat', 
                'Zsym', 'Zsat']
    features_names = [r'$L_{sym}$ (MeV)', r'$K_{sym}$ (MeV)', r'$K_{sat}$ (MeV)', 
                      r'$Z_{sym}$ (MeV)', r'$Z_{sat}$ (MeV)']

    pdf_name = sys.argv[1]
    for filename in sys.argv[2:]:
      with AnalyzeGenData(filename) as analyzer:
        for new_df, weight in analyzer.ReasonableData(features): 
          if g is None:
            mean = analyzer.new_mean
            sd = analyzer.new_sd
            bounds = [[mean[feature]-2*sd[feature], mean[feature]+2*sd[feature]] for feature in features]
            g = fhist.FillablePairGrid(new_df, 
                                       weights=weight['PosteriorWeight'], 
                                       x_names=features_names, 
                                       y_names=features_names,
                                       x_ranges=bounds,
                                       y_ranges=bounds)
            g.map_lower(fhist.FillableHist2D, bins=100, cmap='Reds')
            g.map_upper(fhist.PearsonCorr, bins=100)
            g.map_diag(fhist.FillableHist, bins=50, normalize=True, color='r')
          else:
            g.Append(new_df, weights=weight['PosteriorWeight'])
    g.Draw()
    plt.subplots_adjust(hspace=0.15, wspace=0.15, bottom=0.15, left=0.15, top=0.95)  
    g.fig.set_size_inches(25,25)
    g.fig.align_labels()#tight_layout()

    for i, name in enumerate(features):
      try:
        xlim = g.axes2d[i, i].get_xlim()
        x = np.linspace(*xlim, 100)
        a, b = (xlim[0] - mean[name])/sd[name], (xlim[1] - mean[name])/sd[name]
        y = truncnorm.pdf(x, a, b, loc=mean[name], scale=sd[name])
        g.axes2d[i, i].plot(x, y, color='b')
      except:
        pass

    plt.savefig(pdf_name)
