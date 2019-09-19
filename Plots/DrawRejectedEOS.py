import matplotlib as mpl
mpl.use('Agg')
from Utilities.EOSLoader import EOSLoader
from Plots.FillableHist import FillableHist2D
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from AddWeight import AnalyzeGenData
import itertools as it

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('This script generates pdf images for rejected EOS')
    print('Input: List of filenames from deformability calculation')
    print('Output: pdf files of the image')
    print(' To use, enter\npython %s pdf_name input1 input2 ....' % sys.argv[0])
  else:
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(7, 10)

    # first draw Ksym vs Lsym
    hist = None
    pdf_name = sys.argv[1]
    with AnalyzeGenData(sys.argv[2]) as analyzer:
      for new_df, weight in analyzer.ReasonableData(['Lsym', 'Ksym']):
        if hist is None:
          hist = FillableHist2D(new_df['Lsym'], new_df['Ksym'],
                                bins=100, cmap='Reds', weights=weight['PosteriorWeight'])
        else:
          hist.Append(new_df['Lsym'], new_df['Ksym'], weights=weight['PosteriorWeight'])
    hist.Draw(ax[0])
    #rect = patches.Rectangle((24, 0), 36, 117, fill=False, ec='white')
    #ax[0].add_patch(rect) 

    loader = EOSLoader(sys.argv[2])
    accepted_params = []
    rejected_params = []
    # draw example EOSs
    loader.SetCut(((4*(loader.Backbone_kwargs['Lsym'] -25) - 100 < loader.Backbone_kwargs['Ksym'])))
    # draw boundary
    x = np.linspace(20, 80, 100)
    ax[0].plot(x, 4*(x-25)-100, zorder=11, color='black')
    # label the first 3
    colors = it.chain(['r', 'b'], it.cycle(['black']))
    color_list = []
    zorders = it.chain([10,9], it.cycle([5]))
    for i in range(0, 50):
      if loader.NoNSEOS(i):
        eos = loader.GetNuclearEOS(i)
        density = np.logspace(np.log(1e-9), np.log(1e2), 1000, base=np.e)
        pressure = eos.GetPressure(density)
        energy = eos.GetEnergyDensity(density) 
        color_list.append(next(colors))
        ax[1].plot(energy, pressure, color=color_list[-1], zorder=next(zorders))
        rejected_params.append(i)
         
    rejected = loader.Backbone_kwargs.iloc[rejected_params]
    ax[0].scatter(rejected['Lsym'], rejected['Ksym'], color=color_list, s=50)
    ax[0].set_xlabel(r'$L_{sym}$ (MeV)')
    ax[0].set_ylabel(r'$K_{sym}$ (MeV)')
    ax[1].set_xlim([1e-1, 5e2])
    ax[1].set_ylim([1e-4, 5e2])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\mathcal{E}$ (MeV/fm$^{3}$)')
    ax[1].set_ylabel(r'P (MeV/fm$^{3}$)')
    plt.tight_layout()
    plt.savefig(pdf_name)
    loader.Close()
