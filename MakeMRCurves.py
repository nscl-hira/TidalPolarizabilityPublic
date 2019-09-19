#!/projects/hira/tsangc/Polarizability/myPy/bin/python -W ignore
import math
import sys
import pickle
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
from multiprocessing import Pool
import tempfile
import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd
import scipy.optimize as opt
import argparse
from functools import partial

import Utilities.ConsolePrinter as cp
import TidalLove.TidalLoveWrapper as wrapper
import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from Utilities.EOSCreator import EOSCreator
from MakeSkyrmeFileBisection import LoadSkyrmeFile
from Utilities.EOSLoader import EOSLoader

OuterCrustDensity = 0.3e-3
SurfacePressure = 1e-8

"""
Print the selected EOS into a file for the tidallove script to run
"""
def MRCurvesForModel(name_and_eos, pressure):
    loader = EOSLoader('Results/test.csv.h5')
    eos, list_tran_density = loader.GetNSEOS(name_and_eos)

    radius = []
    mass = []
    with wrapper.TidalLoveWrapper(eos) as tidal_love:
        tidal_love.density_checkpoint = list_tran_density
        for pc in pressure:
            try:
                result = tidal_love.Calculate(pc=pc)
                m, r = result.mass, result.Radius
                if r > 5:
                    mass.append(m)
                    radius.append(r)
            except RuntimeError as error:
                radius.append(np.nan)
                mass.append(np.nan)
    return mass, radius

def FindIntersection(all_results, mass):
    designated_radius = []
    for result in all_results:
        idx = (np.abs(np.array(result[0]) - mass)).argmin()
        designated_radius.append(result[1][idx])
    return min(designated_radius), max(designated_radius)

if __name__ == '__main__':
    #df = LoadSkyrmeFile('Results/ABrownF1Den.csv')
    loader = EOSLoader('Results/test.csv.h5')
    #df = pd.concat([df, LoadSkyrmeFile('Results/ABrownNewNoPolyTrope.csv')])

    #pressure = np.concatenate((np.linspace(2, 500, 50), np.linspace(500, 5000, 100)), axis=None)
    pressure = np.logspace(np.log(1.), np.log(1000), num=50, base=np.exp(1))
    labelq = r'*w.den'
    labelu = r'*v.den'
    q_arglist = [(i, pressure) for i, (name, row) in enumerate(loader.Backbone_kwargs.iterrows()) if name[-1] == 'w']
    u_arglist = [(i, pressure) for i, (name, row) in enumerate(loader.Backbone_kwargs.iterrows()) if name[-1] == 'v']
      
    with Pool(processes=10) as pool:
         q_result = pool.starmap(MRCurvesForModel, q_arglist)
         u_result = pool.starmap(MRCurvesForModel, u_arglist)


    for result in q_result:
         plt.plot(result[1], result[0], color='steelblue', label=labelq, linestyle='--')
         labelq = None
    for result in u_result:
         plt.plot(result[1], result[0], color='orange', label=labelu)
         labelu = None

    # draw verticl line corresponds to 1.4 solar mass NS
    highlighted_mass = 1.4
    plt.axhline(y=highlighted_mass, color='grey', linestyle='-', alpha=0.2)
    q_min, q_max = FindIntersection(q_result, highlighted_mass)
    u_min, u_max = FindIntersection(u_result, highlighted_mass)
    plt.axvspan(q_min, q_max, color='steelblue', alpha=0.2, lw=0)
    plt.axvspan(u_min, u_max, color='orange', alpha=0.2, lw=0)

    plt.xlabel('Radius (km)')
    plt.ylabel(r'Mass ($M_{\odot}$)')
    plt.xlim([7.5, 16])
    plt.legend()
    plt.show()
