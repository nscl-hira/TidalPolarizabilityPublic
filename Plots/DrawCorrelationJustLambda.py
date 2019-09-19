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
    x_features = ['Lsym', 'Ksym', 
                  'P(2rho0)', 'Ksat',
                  'Qsym', 'Qsat']
    x_features_names = [r'$L_{sym}$ (MeV)', r'$K_{sym}$ (MeV)', 
                        r'$P(2\rho_0)$ (MeV/fm$^{3}$)', r'$K_{sat}$ (MeV)', 
                        r'$Q_{sym}$ (MeV)', r'$Q_{sat}$ (MeV)']

    y_features = ['Mass1.2 Lambda', 'Mass1.4 Lambda', 'Mass1.6 Lambda']
    y_features_names = [r'$\Lambda(1.2)$', r'$\Lambda(1.4)$', r'$\Lambda(1.6)$']

    pdf_name = sys.argv[1]
    for filename in sys.argv[2:]:
      with AnalyzeGenData(filename) as analyzer:
        for new_df, weight in analyzer.ReasonableData(x_features + y_features):
          if g is None:
            mean = analyzer.new_mean
            sd = analyzer.new_sd
            x_bounds = [[mean[name]-2*sd[name], mean[name]+2*sd[name]] if name in mean else [10, 50] for name in x_features]
            g = fhist.FillablePairGrid(new_df, 
                                       weights=weight['PosteriorWeight'], 
                                       x_vars=x_features,
                                       x_names=x_features_names, 
                                       x_ranges=x_bounds,
                                       y_vars=y_features,
                                       y_names=y_features_names,
                                       y_ranges=[[1000, 2000], [250, 800], [100, 400]])
            g.map(fhist.FillableHist2D, bins=100, cmap='Reds')
          else:
            g.Append(new_df, weights=weight['PosteriorWeight'])
    g.Draw(fontsize=30)
    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.2, left=0.1, top=0.95)  
    g.fig.set_size_inches(25,10)
    g.fig.align_labels()#tight_layout()

    print('name\tmean\tSD')
    for i, name in enumerate(y_features):
      print('%s\t%f\t%f' % (name, g.graphs[i][0].GetMean(1), g.graphs[i][0].GetSD(1)))

    plt.savefig(pdf_name)
    """
    ax = plt.axes()
    g.graphs[1][2].Draw(ax)
    plt.show()
    """
