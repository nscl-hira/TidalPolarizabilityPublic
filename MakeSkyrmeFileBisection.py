#!/usr/bin/python -W ignore
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

import Utilities.ConsolePrinter as cp
import TidalLove.TidalLoveWrapper as wrapper
import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from EOSCreator import EOSCreator
from SelectPressure import AddPressure

OuterCrustDensity = 0.3e-3
SurfacePressure = 1e-10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--Output", default="Result", help="Name of the CSV output (Default: Result)")
    parser.add_argument("-et", "--EOSType", default="EOS", help="Type of EOS. It can be: EOS, EOSNoPolyTrope, BESkyrme, OnlySkyrme (Default: EOS)")
    parser.add_argument("-sd", "--SkyrmeDensity", type=float, default=0.3, help="Density at which Skyrme takes over from crustal EOS (Default: 0.3)")
    parser.add_argument("-pp", "--PolyTropeDensity", type=float, default=3, help="Density at which Skyrme EOS ends. (Default: 3)")
    parser.add_argument("-td", "--TranDensity", type=float, default=0.001472, help="Density at which Crustal EOS ends (Default: 0.001472)")
    parser.add_argument("-pd", "--PRCDensity", type=float, default=None, help="Enable PRC automatic density transition. Value entered determine fraction of density that is represented by relativistic gas")
    parser.add_argument("-cs", "--CrustSmooth", type=float, default=0, help="degrees of smoothing. Reduce oscillation of speed of sound near crustal volumn")
    parser.add_argument("-mm", "--MaxMass", type=float, default=2, help="Maximum Mass to be achieved for EOS in unit of solar mass (Default: 2)")
    args = parser.parse_args()

    df = pd.read_csv('SkyrmeParameters/PawelSkyrmeNew.csv', index_col=0)
    df.fillna(0, inplace=True)

    summary = sky.SummarizeSkyrme(df)

    title = ['Model', 'R(1.4)', 'lambda(1.4)']
    if args.EOSType == 'EOS': 
        title = title + ['SDToRTDRadius', 'RTDToRadius', 'OutCrustRad']
    printer = cp.ConsolePrinter(title)
    
    """
    Print the selected EOS into a file for the tidallove script to run
    """
    def CalculateModel(name_and_eos):
        name = name_and_eos[0]    
        eos_creator = EOSCreator(name_and_eos[1], 
                                 TranDensity=args.TranDensity*rho0, 
                                 SkyrmeDensity=args.SkyrmeDensity*rho0, 
                                 PolyTropeDensity=args.PolyTropeDensity*rho0, 
                                 PRCTransDensity=args.PRCDensity, 
                                 CrustSmooth=args.CrustSmooth)
        """
        Depending on the type of EOS, different calculation is performed
        if EOSType == EOS, it will calculate pressure at 7rho0 such that max mass = 2
        and the corresponding central pressure

        For all other EOS, it will just calculate max mass and 1.4 Neutron star
        """

        """
        Prepare EOS
        """
        pressure_high = 500.
        pc_max = [0]
        max_mass = args.MaxMass
        if(args.EOSType == "EOS"):
            pressure_high, pc_max = eos_creator.PrepareEOS(args.EOSType, max_mass=max_mass)
        eos, list_tran_density = eos_creator.GetEOSType(args.EOSType)
        tidal_love = wrapper.TidalLoveWrapper(eos)
        if(args.EOSType != "EOS"):
            max_mass, pc_max = tidal_love.FindMaxMass()

        """
        1.4 solar mass calculation
        """
        tidal_love.checkpoint = np.append(eos.GetAutoGradPressure(np.array(list_tran_density + [OuterCrustDensity]), 0), [SurfacePressure])
        try:
            mass, radius, lambda_, pc14, checkpoint_mass, checkpoint_radius = tidal_love.FindMass(mass=1.4)
            _, _, _, pc2, _, _ = tidal_love.FindMass(mass=2., central_pressure0=600)
        except RuntimeError as error:
            mass = np.nan
            radius = np.nan 
            lambda_ = np.nan
            pc14 = np.nan
        tidal_love.Close()
        result = {'Model':name, 'R(1.4)':radius, 'lambda(1.4)':lambda_, 'PCentral':pc14, 'PCentral2MOdot': pc2, 'PCentralMaxMass':pc_max[0], 'MaxMass': max_mass}
        if(args.EOSType == "EOS"):
            result['PolyHighP'] = pressure_high
            result['SDToRTDRadius'] = checkpoint_radius[2] - checkpoint_radius[1]
            result['RTDToRadius'] = radius - checkpoint_radius[2]
            result['OutCrustRad'] = radius - checkpoint_radius[3]
        printer.PrintContent(result)

        return result


    name_list = [(index, sky.Skryme(row)) for index, row in df.iterrows()]

    result = []
    with ProcessPool() as pool:
        future = pool.map(CalculateModel, name_list, timeout=120)
        iterator = future.result()
        while True:
            try:
                result.append(next(iterator))
            except StopIteration:
                break
            except TimeoutError as error:
                pass
                #print("function took longer than %d seconds" % error.args[1])
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process
            sys.stdout.flush()

    data = [val for val in result]
    data = pd.DataFrame.from_dict(data)
    data.set_index('Model', inplace=True)
    data = pd.concat([df, summary, data], axis=1)
    data.dropna(axis=0, how='any', inplace=True)
    data = AddPressure(data)
    data.to_csv('Results/%s.csv' % args.Output, index=True)
