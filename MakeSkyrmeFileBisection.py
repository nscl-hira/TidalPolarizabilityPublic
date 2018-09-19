#!/projects/hira/tsangc/Polarizability/myPy/bin/python -W ignore
import math
import sys
import cPickle as pickle
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

OuterCrustDensity = 0.3e-3
SurfacePressure = 1e-8

def LoadSkyrmeFile(filename):
    df = pd.read_csv(filename, index_col=0)
    return df.fillna(0)

"""
Print the selected EOS into a file for the tidallove script to run
"""
def CalculateModel(name_and_eos, **kwargs):
    name = name_and_eos[0]    
    EOSType = kwargs['EOSType']
    max_mass = kwargs['MaxMassRequested']
    eos_creator = EOSCreator(name_and_eos[1], **kwargs)


    """
    Prepare EOS
    """
    additional_para = eos_creator.PrepareEOS(EOSType, max_mass=max_mass)
    eos, list_tran_density = eos_creator.GetEOSType(EOSType)

    # insert surface density
    list_tran_density.append(OuterCrustDensity)

    """
    Bill asked for what happens at rho0 and 2rho0
    """
    rho0 = 0.16
    list_tran_density.append(rho0)
    list_tran_density.append(2.*rho0)

    """
    list of transition density must be in desending order...
    Need to sort it
    """
    list_tran_density.sort(reverse=True)


    """
    1.4 solar mass and 2.0 solar mass calculation
    """
    tidal_love = wrapper.TidalLoveWrapper(eos, 'EOS_%s' % name)
    max_mass, pc_max = tidal_love.FindMaxMass()
    tidal_love.checkpoint = np.append(eos.GetAutoGradPressure(np.array(list_tran_density), 0), [SurfacePressure])
    try:
        mass, radius, lambda_, pc14, checkpoint_mass, checkpoint_radius = tidal_love.FindMass(mass=1.4)
        _, _, _, pc2, _, _ = tidal_love.FindMass(mass=2., central_pressure0=300)
    except RuntimeError as error:
        tidal_love.Close()
        raise ValueError('Failed to find 1.4/2.0 solar mass properties for this EOS')
    if mass < 1e-4 or lambda_ < 1e-4:
        tidal_love.Close()
        raise ValueError('Mass/Lambda = 0. Calculation failed')
    if any(np.isnan([mass, radius, lambda_, pc14])) or any(np.isnan(checkpoint_mass)) or any(np.isnan(checkpoint_radius)):
        tidal_love.Close()
        raise ValueError('Some of the calculated values are nan.')


    """
    Write results to dict and return
    """
    result = {'Model':name, 
              'R(1.4)':radius, 
              'lambda(1.4)':lambda_, 
              'PCentral':pc14, 
              'PCentral2MOdot': pc2, 
              'PCentralMaxMass':pc_max, 
              'MaxMass': max_mass} 
    for den, (index, radius) in zip(list_tran_density, enumerate(checkpoint_radius)):
        result['RadiusCheckpoint%d' % index] = radius
        result['DensityCheckpoint%d' % index] = den
    for key, val in kwargs.iteritems():
        result[key] = val
    for key, val in additional_para.iteritems():
        result[key] = val

    return result



def CalculatePolarizability(df, Output, **kwargs):
    summary = sky.SummarizeSkyrme(df)
    EOSType = kwargs['EOSType']

    """
    Tells ConsolePrinter which quantities to be printed in real time
    """
    title = ['Model', 'R(1.4)', 'lambda(1.4)', 'PCentral']
    printer = cp.ConsolePrinter(title)
    
    
    """
    Create multiple pools for parallel computation
    """
    name_list = [(index, row) for index, row in df.iterrows()]
    result = []
    with ProcessPool() as pool:
        future = pool.map(partial(CalculateModel, **kwargs), name_list, timeout=60)
        iterator = future.result()
        while True:
            try:
                new_result = next(iterator)
                result.append(new_result)
                printer.PrintContent(new_result)
            except StopIteration:
                break
            except ValueError as e:
                pass
            except TimeoutError as error:
                pass
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process

            sys.stdout.flush()

    """
    Merge calculation data with Skyrme loaded data
    """
    data = [val for val in result]
    data = pd.DataFrame.from_dict(data)
    data.set_index('Model', inplace=True)
    data = pd.concat([df, summary, data], axis=1)
    #data = pd.concat([df, data], axis=1)    
    data.dropna(axis=0, how='any', inplace=True)

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--Input", default="SkyrmeParameters/PawelSkyrme.csv", help="Name of the Skyrme input file (Default: SkyrmeResult/PawelSkyrme.csv)")
    parser.add_argument("-o", "--Output", default="Result", help="Name of the CSV output (Default: Result)")
    parser.add_argument("-et", "--EOSType", default="EOS", help="Type of EOS. It can be: EOS, EOSNoPolyTrope, BESkyrme, OnlySkyrme (Default: EOS)")
    parser.add_argument("-sd", "--SkyrmeDensity", type=float, default=0.3, help="Density at which Skyrme takes over from crustal EOS (Default: 0.3)")
    parser.add_argument("-pp", "--PolyTropeDensity", type=float, default=3, help="Density at which Skyrme EOS ends. (Default: 3)")
    parser.add_argument("-td", "--TranDensity", type=float, default=0.001472, help="Density at which Crustal EOS ends (Default: 0.001472)")
    parser.add_argument("-pd", "--PRCTransDensity", type=float, default=None, help="Enable PRC automatic density transition. Value entered determine fraction of density that is represented by relativistic gas")
    parser.add_argument("-cs", "--CrustSmooth", type=float, default=0, help="degrees of smoothing. Reduce oscillation of speed of sound near crustal volumn")
    parser.add_argument("-mm", "--MaxMassRequested", type=float, default=2, help="Maximum Mass to be achieved for EOS in unit of solar mass (Default: 2)")
    args = parser.parse_args()

    df = LoadSkyrmeFile(args.Input)
    argd = vars(args)
    argd['TranDensity'] = argd['TranDensity']*rho0
    argd['SkyrmeDensity'] = argd['SkyrmeDensity']*rho0
    argd['PolyTropeDensity'] = argd['PolyTropeDensity']*rho0

    df = CalculatePolarizability(df, **argd)
    df = AddPressure(df)
    df.to_csv('Results/%s.csv' % args.Output, index=True)

