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

import TidalLove.TidalLoveWrapper as wrapper
import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from EOSCreator import EOSCreator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--Output", default="Result", help="Name of the CSV output (Default: Result)")
    parser.add_argument("-et", "--EOSType", default="EOS", help="Type of EOS. It can be: EOS, EOSNoPolyTrope, BESkyrme, OnlySkyrme (Default: EOS)")
    parser.add_argument("-sd", "--SkyrmeDensity", type=float, default=0.3, help="Density at which Skyrme takes over from crustal EOS (Default: 0.3)")
    parser.add_argument("-td", "--TranDensity", type=float, default=0.001472, help="Density at which Crustal EOS ends (Default: 0.001472)")
    parser.add_argument("-pd", "--PRCDensity", type=float, default=None, help="Enable PRC automatic density transition. Value entered determine fraction of density that is represented by relativistic gas")
    args = parser.parse_args()

    df = pd.read_csv('SkyrmeParameters/PawelSkyrme.csv', index_col=0)
    df.fillna(0, inplace=True)

    summary = sky.SummarizeSkyrme(df)

    title='MaxMass'
    if args.EOSType == "EOS":
        title='PressureHigh'

    print('''\
{dashes}
{m:^12} | {r:^12} | {l:^12} | {p:^12} | {a:^12} | {b:^12} | {c:^12}
{dashes}'''.format(dashes='-'*100, m='name', r='radius1.4', l='lambda1.4', p='pc1.4', a='pcMaxMass', b=title, c='Causal'))
    
    """
    Print the selected EOS into a file for the tidallove script to run
    """
    def CalculateModel(name_and_eos):
        name = name_and_eos[0]    
        eos_creator = EOSCreator(name_and_eos[1], TranDensity=args.TranDensity*rho0, SkyrmeDensity=args.SkyrmeDensity*rho0, PRCTransDensity=args.PRCDensity)

        """
        Depending on the type of EOS, different calculation is performed
        if EOSType == EOS, it will calculate pressure at 7rho0 such that max mass = 2
        and the corresponding central pressure

        For all other EOS, it will just calculate max mass and 1.4 Neutron star
        """
  
        if(args.EOSType == "EOS"):
            def FixMaxMass(pressure_high):
                eos_creator.PressureHigh = pressure_high
                eos, _ = eos_creator.GetEOSType(args.EOSType)
                tidal_love = wrapper.TidalLoveWrapper(eos)
                
                max_mass, pc_max = tidal_love.FindMaxMass()
                tidal_love.Close()
                return max_mass - 2.

            try:
                pressure = opt.newton(FixMaxMass, x0=800)
                eos_creator.PressureHigh = pressure
                eos, list_tran_density = eos_creator.GetEOSType(args.EOSType)
                print('tran dens', list_tran_density, eos.GetAutoGradPressure(np.array(list_tran_density), 0))
                tidal_love = wrapper.TidalLoveWrapper(eos)
                tidal_love.checkpoint = eos.GetAutoGradPressure(np.array(list_tran_density), 0)
                print('pressure', tidal_love.checkpoint)
                mass, radius, lambda_, pc14, checkpoint_mass, checkpoint_radius = tidal_love.FindMass14()
                print('mass ', mass, ' radius ', radius, ' checkpoint mass ', checkpoint_mass, ' checkpoint radius ', checkpoint_radius)
                max_mass, pc_max = tidal_love.FindMaxMass()
                tidal_love.Close()

                # look for pressure corresponding the central pressure of max mass
                # Then check for causailiry in range
                highest_density = opt.newton(lambda rho: eos.GetAutoGradPressure(rho,0), x0=7*0.16)
                causal = not eos.ViolateCausality(np.linspace(1e-5, highest_density, 100), 0)
            except RuntimeError as error:
                pressure = np.nan
                mass, radius, lambda_, causal = np.nan, np.nan, np.nan, np.nan

            
      
    
            if math.isnan(sum([mass, radius, lambda_, pressure])):
                mass, radius, lambda_, pressure, pc_max = np.nan, np.nan, np.nan, np.nan, [np.nan]
            else:
                print("{m:^12} | {r:^12.3f} | {l:^12.3f} | {p:^12.3f} | {a:^12.3f} | {b:^12.3f} | {c:^12}".format(m=name, r=radius, l=lambda_, p=pc14, a=pc_max[0], b=pressure, c=causal))

            return name, mass, radius, lambda_, pc14, pc_max[0], pressure, causal

        else:
            try:
                eos = eos_creator.GetEOSType(args.EOSType)
                tidal_love = wrapper.TidalLoveWrapper(eos)
                mass, radius, lambda_, pc14 = tidal_love.FindMass14()
                max_mass, pc_max = tidal_love.FindMaxMass()
                tidal_love.Close()

                # look for pressure corresponding the central pressure of max mass
                # Then check for causailiry in range
                highest_density = opt.newton(lambda rho: eos.GetAutoGradPressure(rho,0), x0=7*0.16)
                causal = not eos.ViolateCausality(np.linspace(1e-5, highest_density, 100), 0)
            except RuntimeError as error:
                mass, radius, lambda_, max_mass, pc_max = np.nan, np.nan, np.nan, np.nan, [np.nan]

            if math.isnan(sum([mass, radius, lambda_, max_mass, pc_max[0]])):
                mass, radius, lambda_, max_mass, pc_max = np.nan, np.nan, np.nan, np.nan, [np.nan]
            else:
                print("{m:^12} | {r:^12.3f} | {l:^12.3f} | {p:^12.3f} | {a:^12.3f} | {b:^12.3f} | {c:^12}".format(m=name, r=radius, l=lambda_, p=pc14, a=pc_max[0], b=max_mass, c=causal))

            return name, mass, radius, lambda_, pc14, pc_max[0], max_mass, causal

    name_list = [(index, sky.Skryme(row)) for index, row in df.iterrows()]

    result = []
    with ProcessPool() as pool:
        future = pool.map(CalculateModel, name_list, timeout=60)
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

    data = [{'Model':val[0], 'R(1.4)':val[2], 'lambda(1.4)':val[3], 'PCentral':val[4], title:val[5]} for val in result]
    data = pd.DataFrame.from_dict(data)
    data.set_index('Model', inplace=True)
    data = pd.concat([df, summary, data], axis=1)
    data.dropna(axis=0, how='any', inplace=True)
    data.to_csv('Results/%s.csv' % args.Output, index=True)
