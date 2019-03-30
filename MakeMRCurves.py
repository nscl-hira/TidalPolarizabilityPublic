#!/projects/hira/tsangc/Polarizability/myPy/bin/python -W ignore
import math
import sys
import pickle
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
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
from SelectPressure import AddPressure
from MakeSkyrmeFileBisection import LoadSkyrmeFile

OuterCrustDensity = 0.3e-3
SurfacePressure = 1e-8

"""
Print the selected EOS into a file for the tidallove script to run
"""
def MRCurvesForModel(name_and_eos, pressure):
    name = name_and_eos[0]    
    eos_creator = EOSCreator(name_and_eos[1])


    """
    Prepare EOS
    """
    kwargs = eos_creator.PrepareEOS(**name_and_eos[1])
    eos, list_tran_density = eos_creator.GetEOSType(**name_and_eos[1])

    # insert surface density
    list_tran_density.append(OuterCrustDensity)
    """
    list of transition density must be in desending order...
    Need to sort it
    """
    list_tran_density.sort(reverse=True)


    radius = []
    mass = []
    with wrapper.TidalLoveWrapper(eos) as tidal_love:
        tidal_love.checkpoint = np.append(eos.GetPressure(np.array(list_tran_density), 0), [SurfacePressure])
        for pc in pressure:
            try:
                m, r, _, _, _ = tidal_love.Calculate(pc=pc)
                if r > 5:
                    mass.append(m)
                    radius.append(r)
            except RuntimeError as error:
                radius.append(np.nan)
                mass.append(np.nan)
    return mass, radius


if __name__ == '__main__':
    df = LoadSkyrmeFile('Results/ABrownUDen.csv')
    df = pd.concat([df, LoadSkyrmeFile('Results/ABrownNewNoPolyTrope.csv')])
    print(df)

    #pressure = np.concatenate((np.linspace(2, 500, 50), np.linspace(500, 5000, 100)), axis=None)
    pressure = np.logspace(np.log(1.), np.log(1000), num=100, base=np.exp(1))
    labelq = 'Group q'
    labelu = 'Group u'
    for name, row in df.iterrows():
      
      if name[-1] == 'q':
         mass, radius = MRCurvesForModel([name, df.loc[name]], pressure)
         plt.plot(radius, mass, color='b', label=labelq, linestyle='--')
         labelq = None
      if name[-1] == 'u':
         mass, radius = MRCurvesForModel([name, df.loc[name]], pressure)
         plt.plot(radius, mass, color='r', label=labelu)
         labelu = None

    plt.xlabel('Radius (km)')
    plt.ylabel(r'Mass ($M_{\odot}$)')
    plt.legend()
    plt.show()
