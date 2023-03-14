import os
import matplotlib.patches as patches
from Utilities.EOSCreator import EOSCreator, NuclearEOSFactory
from Plots.FillableHist import FillableHist2D
import numpy as np
import pandas as pd
import sys

from AddWeight import AnalyzeGenData

def UnrollMeta(meta, i):
  columns = list(meta.columns.levels[0])
  meta_dict = {}
  for col in columns:
    meta_dict[col] = meta[col].iloc[i].values
  return meta_dict

class EOSLoader:

  def __init__(self, filename, start=0, end=None):
    self.store = pd.HDFStore(filename, 'r')
    length = self.store.get_storer('kwargs').nrows
    if end is None:
        end = length

    head, ext = os.path.splitext(filename)
    self.weight_store = None
    if os.path.exists(head + '.Weight' + ext):
      self.weight_store = pd.HDFStore(head + '.Weight' + ext, mode='r')
      self.reasonable = self.weight_store.select('main', start=start, stop=end)['Reasonable']
      self.weight = self.weight_store.select('main', start=start, stop=end)['PosteriorWeight']

    self.meta_data = self.store.select('meta', start=start, stop=end)
    self.Backbone_kwargs = self.store.select('kwargs', start=start, stop=end)
    self.Transform_kwargs = self.store.get_storer('kwargs').attrs.meta_data
    if 'MaxMassRequested' in self.Transform_kwargs:
      self.Transform_kwargs['MaxMass'] = self.Transform_kwargs['MaxMassRequested']
    self.EOSType = self.Transform_kwargs['EOSType']
    self.NoData = self.store.select('EOSCheck', start=start, stop=end)['NoData']
    self.Causality = self.store.select('EOSCheck', start=start, stop=end)['ViolateCausality']
    self.creator = EOSCreator()

  def __len__(self):
    return self.meta_data.shape[0]

  def SetCut(self, cut):
    self.meta_data = self.meta_data[cut]
    self.Backbone_kwargs = self.Backbone_kwargs[cut]
    self.NoData = self.NoData[cut]
    self.Causality = self.Causality[cut]

  def NoNSEOS(self, i):
    return (self.NoData.iloc[i] & (not self.Causality.iloc[i]))

  def GetNuclearEOS(self, i):
    return NuclearEOSFactory(self.EOSType, self.Backbone_kwargs.iloc[i])

  def GetNSEOS(self, i):
    eos, density_list, _ = self.creator.Factory(self.EOSType, 
                                                self.Backbone_kwargs.iloc[i],
                                                self.Transform_kwargs,
                                                UnrollMeta(self.meta_data, i))
    return eos, [1e-9] + density_list + [10*0.16]

  def GetName(self, i):
    return self.Backbone_kwargs.index[i]

  def Close(self):
    self.store.close()
    if self.weight_store is not None:
      self.weight_store.close()



"""
if __name__ == '__main__':
  fig, ax = plt.subplots(nrows=2, ncols=1)
  fig.set_size_inches(7, 10)

  # first draw Ksym vs Lsym
  hist = None
  with AnalyzeGenData('Results/SmoothSound2.Gen.h5') as analyzer:
    for new_df, weight in analyzer.ReasonableData(['Lsym', 'Ksym']):
      if hist is None:
        hist = FillableHist2D(new_df['Lsym'], new_df['Ksym'],
                              bins=100, cmap='inferno', weights=weight['PosteriorWeight'])
      else:
        hist.Append(new_df['Lsym'], new_df['Ksym'], weights=weight['PosteriorWeight'])
  hist.Draw(ax[0])
  rect = patches.Rectangle((24, 0), 36, 117, fill=False, ec='white')
  ax[0].add_patch(rect) 

  loader = EOSLoader('Results/SmoothSound2.Gen.h5')
  accepted_params = []
  rejected_params = []
  # draw example EOSs
  loader.SetCut(((loader.Backbone_kwargs['Lsym'] < 60) & (loader.Backbone_kwargs['Ksym'] > 0)))
  for i in range(0, 50):
    if loader.NoNSEOS(i):
      eos = loader.GetNuclearEOS(i)
      density = np.logspace(np.log(1e-9), np.log(1e2), 1000, base=np.e)
      pressure = eos.GetPressure(density)
      energy = eos.GetEnergyDensity(density) 
      ax[1].plot(energy, pressure, color='black')
      rejected_params.append(i)
       
  rejected = loader.Backbone_kwargs.iloc[rejected_params]
  ax[0].scatter(rejected['Lsym'], rejected['Ksym'], color='white')
  ax[0].set_xlabel(r'$L_{sym}$')
  ax[0].set_ylabel(r'$K_{sym}$')
  ax[1].set_xlim([1e-1, 5e2])
  ax[1].set_ylim([1e-4, 5e2])
  ax[1].set_xscale('log')
  ax[1].set_yscale('log')
  ax[1].set_xlabel(r'E (MeV/fm$^{3}$)')
  ax[1].set_ylabel(r'P (MeV/fm$^{3}$)')
  plt.tight_layout()
  plt.show()

  loader.Close()
"""
