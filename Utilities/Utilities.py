import tempfile
import os
import itertools
color = itertools.cycle(('fuchsia', 'r', 'r', 'b', 'g', 'orange')) 
#marker = itertools.cycle(('o','v','^','*','s')) 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as font_manager

#font_dirs = ['/projects/hira/tsangc/Polarizability/fonts', ]
#font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#
#for font_file in font_files:
#  font_list = font_manager.FontManager.addfont(font_file)
#font_list = font_manager.createFontList(font_files)
#font_manager.fontManager.ttflist.extend(font_list)

from multiprocessing_logging import install_mp_handler, MultiProcessingHandler
import shutil
import logging
logger = logging.getLogger(__name__)
install_mp_handler(logger)




import matplotlib.pylab as pylab
params = {#'figure.autolayout': True,
          'figure.figsize': [8.61, 7],
          'legend.fontsize': 20,
          'legend.framealpha': 0,
          'lines.linewidth': 2,
          'lines.markeredgewidth': 2,
          'lines.markersize': 10,
          'errorbar.capsize': 3,
          #'font.family': 'stixgeneral', #'serif',
          #'font.serif': 'stix',
          'font.family': 'serif',
          'font.size': 25,
          #'font.serif': 'CMU Typewriter Text', #'cmr12', 
          #'text.usetex': True,
          #'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'mathtext.default': 'regular',
          #'patch.linewidth': 0,
          'axes.linewidth': 2,
          'axes.labelsize': 30,
          #'axes.labelsize': 35,
          #'axes.labelpad': 15,
          'figure.subplot.bottom': 0.15,
          'figure.subplot.left': 0.20,
          'axes.unicode_minus': True,
          'xtick.labelsize':25,
          'ytick.labelsize':25,
          'xtick.major.size': 7,
          'xtick.minor.size': 3,
          'xtick.major.width': 2,
          'xtick.minor.width': 2,
          'xtick.minor.visible': True,
          'xtick.major.pad': 10,
          'xtick.direction': 'in',
          'ytick.major.size': 7,
          'ytick.minor.size': 3,
          'ytick.major.width': 2,
          'ytick.minor.width': 1.5,
          'ytick.minor.visible': True,
          'ytick.direction': 'in',
          'ytick.major.pad': 10,
          'figure.facecolor': 'white'}
pylab.rcParams.update(params)


import numpy as np
import Utilities.SkyrmeEOS as sky
from Utilities.Constants import *

def FlattenListElements(df):
    df_list = []
    for key in df.columns:
        if isinstance(df[key].iat[0], (list, np.ndarray)):
            values = np.vstack(df[key])
            if values.shape[1] > 1:
                df_list.append(pd.DataFrame(values, index=df.index))
            else:
                df_list.append(pd.DataFrame(values, index=df.index, columns=[key]))
        else:
            df_list.append(df[key])
    new_df = pd.concat(df_list, keys=df.columns, axis=1)
    #new_df.index = df.index
    return new_df



def ConcatenateListElements(df):
    df_list = []
    for key in set(df.columns.get_level_values(0)):
        if len(df[key].shape) == 1:
            df_list.append(df[key])
        elif df[key].shape[1] < 2:
            df_list.append(df[key])
        else:
            df_list.append(pd.Series(tuple(df[key].values), name=key, index=df.index))
    return pd.concat(df_list, axis=1)

def GetContour(df, rho_min, rho_max):
    n = np.linspace(rho_min,rho_max,1000)
    value = []
    for index, row in df.iterrows():
        eos = sky.Skryme(row)
        value.append(eos.GetEnergy(n*eos.rho0, 0.))
    value = np.array(value)
    contour = np.concatenate([np.amax(value, axis=0), np.amin(value, axis=0)[::-1]]).flatten()
    values = np.array([n, n[::-1]]).flatten()
    # complete the loop by including the first element as the last
    contour = np.append(contour, contour[0])
    values = np.append(values, values[0])
    return values, contour

class DataIO:
    def __init__(self, filename, flush_interval=10):
        self.flush_interval = flush_interval
        self.filename = filename
        self.store = pd.HDFStore(self.filename, 'a')
        # names: Index of a table for each branch
        # Should be the same for all branches, but different branch flush at different time so it is more convinent ot just save names for each branch
        self.names = {}
        # values: Value of each branch. DataFrame will be made out of it
        self.values = {}
        if flush_interval <= 0:
            logger.error('Invalid flush_interval. It must be larger than zero.')


    def AppendData(self, branch, name, value, second_lvl_branch=None):
        # names can only be 30 characters long
        name = name[:30]
        if second_lvl_branch is None:
            if branch not in self.names:
                self.names[branch] = []
                self.values[branch] = []
            if len(self.names[branch]) == self.flush_interval:
                self.Flush(branch)

            self.names[branch].append(name)
            self.values[branch].append(value)
        else:
            if branch not in self.names:
                self.names[branch] = {}
                self.values[branch] = {}
            if second_lvl_branch not in self.values[branch]:
                self.names[branch][second_lvl_branch] = []
                self.values[branch][second_lvl_branch] = []
            # need to flush before appending data
            # such that the list of second_lvl_branch can be completed before flushing
            if len(self.names[branch][second_lvl_branch]) == self.flush_interval:
                self.Flush(branch)
            self.names[branch][second_lvl_branch].append(name)
            self.values[branch][second_lvl_branch].append(value)
            # Warning: Flush won't always flush second level branch unless each second level branch has the same length
           
    def AppendMeta(self, branch, meta_data):
        if branch not in self.store.keys():
            self.Flush(branch)
        self.store.get_storer(branch).attrs.meta_data = meta_data

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.Close()

    def Close(self):
        self.Flush()
        self.store.close()

    def Flush(self, branches=None):
        if branches is None:
            branches = self.names.keys()
        # need to loop through branches even if it is just one. Make it list if branches is a string
        elif isinstance(branches, str):
            branches = [branches]
        # store the supplied branch
        for branch in branches:
            data = None
            # this branch has second level if the content is a dictionary
            if isinstance(self.values[branch], dict):
                # flush only when all second level branches reaches the same length
                # because we need to combine all second level branches into one dataframe. 
                # if the length of second lvl branches are not aligned, the dataframe won't be rectanglar
                if len(set(len(value) for title, value in self.values[branch].items())) == 1: # won't even save if the list is empty
                    # create dataframe for each second lvl branch and concat at the end
                    dfs = {}
                    for title, value in self.values[branch].items():
                        if len(self.names[branch][title]) > 0:
                            temp_df = pd.DataFrame.from_dict(value)
                            temp_df.index = self.names[branch][title]
                            temp_df = FlattenListElements(temp_df)

                            dfs[title] = temp_df
                        self.names[branch][title] = []
                        self.values[branch][title] = []
                    if len(dfs) > 0:
                        data = pd.concat(dfs, axis=1)
                else:
                    logger.warning('Lengths of the second lvl branches are not identical. Will not flush')
            elif len(self.names[branch]) > 0:
                data = pd.DataFrame.from_dict(self.values[branch])
                data.index = self.names[branch]
                data = FlattenListElements(data)

                self.names[branch] = []
                self.values[branch] = []
            if data is not None:
                self.store.append(branch, data, min_itemsize={'index': 30})


    

def PlotSkyrmeSymEnergy(df, ax, range_=[0,3], pfrac=0, label=None, **args):
    n = np.linspace(*range_, num=1000)
    first = True
    for index, row in df.iterrows():
        eos = sky.Skryme(row)
        if not first:
            label = None
        first = False
        ax.plot(n, eos.GetAsymEnergy(n*eos.rho0), label=label, zorder=-32, **args)
        
    ax.set_ylabel(r'$S(\rho) (MeV)}$')
    ax.set_xlabel(r'$Density\ \rho/\rho_{0}$')
    return ax

def PlotSkyrmeEnergy(df, ax, range_=[0,3], pfrac=0, label=None, **args):
    n = np.linspace(*range_, num=1000)
    first = True
    for index, row in df.iterrows():
        eos = sky.Skryme(row)
        energy = eos.GetEnergy(n*eos.rho0, pfrac) - mn
        if not first:
            label = None
        first = False
        labels = ax.plot(n, energy, label=label, **args)
    ax.set_xlabel('$Density\ \\rho/\\rho_{0}$')
    ax.set_ylabel('Energy per nucleons (MeV)')
    return ax

def PlotSkyrmePressure(df, ax, range_=[0,3], pfrac=0, label=None, **args):
    n = np.linspace(*range_, num=1000)
    first = True
    for index, row in df.iterrows():
        eos = sky.Skryme(row)
        pressure = eos.GetPressure(n*eos.rho0, pfrac)
        if not first:
            label = None
        first = False
        ax.plot(n, pressure, label=label, zorder=1, **args)
    ax.set_ylim([1,1000])
    ax.set_xlim(range_)
    #ax.set_yscale('log')
    ax.set_xlabel('$Density\ \\rho/\\rho_{0}$')
    ax.set_ylabel('$Pressure (MeV/fm^{3})$')
    return ax

def PlotSkyrmePressureEnergy(df, ax, range_=[0,3], pfrac=0, label=None, **args):
    n = np.linspace(*range_, num=1000)
    first = False
    for index, row in df.iterrows():
        eos = sky.Skryme(row)
        energy = (eos.GetEnergy(n, pfrac) + mn)*(n)
        pressure = eos.GetPressure(n, pfrac)
        if not first:
            label = None
        first = False
        labels = ax.plot(energy, pressure, label=label, zorder=1, **args)
    ax.set_ylim([0.5,1e4])
    ax.set_xlim([1e2, 3e4])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('$Energy\ density\ (MeV/fm^{3})$')
    ax.set_ylabel('$Pressure (MeV/fm^{3})$')
    return labels

def PlotMassVsRadius(mass, radius, ax, **args):
    for key in mass:
        ax.plot(radius[key], mass[key], label=None, **args)
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Mass (solar)')
    ax.set_ylim([0, 3])
    ax.set_xlim([0, 20])
    return ax

def PlotMassVsLambda(mass, lambda_, ax, **args):
    for key in mass:
        ax.plot(mass[key], lambda_[key], label=None, **args)
    ax.set_xlabel('Mass (solar)')
    ax.set_ylabel('lambda')
    ax.set_ylim([0, 4000])
    return ax

def PlotLambdaRadius(mass, radius, lambda_, ax, **args):
    for key in mass:
        ax.plot([np.interp(1.4, mass[key], radius[key])], [np.interp(1.4, mass[key], lambda_[key])], label=None, marker=marker.next(), markersize=20, **args)
    ax.set_xlabel('R(1.4 solar mass) km')
    ax.set_ylabel('lambda')
    ax.set_ylim([-200, 2000])
    ax.set_xlim([0, 20])
    return ax
    
def PlotMaster(df, constrainted_df_list, labels, color_list=('royalblue', 'g', 'orange'), pfrac=0):
    fig = plt.figure(figsize=(18,11))
    ax1 = fig.add_subplot(121)
    

    # also plot all for comparison for symmetry term
    #ax1 = PlotSkyrmeSymEnergy(df, ax1, color='lawngreen', range_=[0,5])
    ax1 = PlotSkyrmeSymEnergy(df, ax1, color='lawngreen', range_=[0,5], pfrac=pfrac)
    color = itertools.cycle(color_list)
    for constrainted_df, label in zip(constrainted_df_list, labels):
        ax1 = PlotSkyrmeSymEnergy(constrainted_df, ax1, pfrac=pfrac, color=color.next(), range_=[0,5], label=label)
    
    #ax1.set_ylim([-20,50])
    #ax1.set_xlim([1., 3.])
    ax1.set_xlim([0, 1.5])
    ax1.set_ylim([0, 50])

    minor_ticks = np.arange(0, 1.5, 0.1)
    major_ticks = np.arange(0, 1.5, 0.5)
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)

    minor_ticks = np.arange(0, 50, 2)
    major_ticks = np.arange(0, 50, 10)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    
    ax2 = fig.add_subplot(122)
    # plot background as comparison
    ax2 = PlotSkyrmePressure(df, ax2, pfrac=pfrac, color='lawngreen', range_=[1e-1,5])
    color = itertools.cycle(color_list)
    for constrainted_df, label in zip(constrainted_df_list, labels):
        ax2 = PlotSkyrmePressure(constrainted_df, ax2, pfrac=pfrac, color=color.next(), range_=[1,5], label=label)
    ax2.set_ylim([1,1300])
    ax2.set_xlim([1,5])
    ax2.set_yscale('log')

    minor_ticks = np.arange(1, 5, 0.2)
    major_ticks = np.arange(1, 5, 1)
    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)

    return ax1, ax2
