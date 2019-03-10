#!/projects/hira/tsangc/Polarizability/myPy/bin/python -W ignore
import sys
from multiprocessing import cpu_count
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
import autograd.numpy as np
import pandas as pd
import argparse
from functools import partial
import scipy.optimize as opt

import Utilities.ConsolePrinter as cp
import TidalLove.TidalLoveWrapper as wrapper
from Utilities.Constants import *
from Utilities.EOSCreator import EOSCreator, SummarizeSkyrme
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
    max_mass_req = kwargs['MaxMassRequested']
    target_mass = kwargs['TargetMass']
    eos_creator = EOSCreator(name_and_eos[1])


    """
    Prepare EOS
    """
    sys.stdout.flush()
    kwargs = eos_creator.PrepareEOS(**kwargs)
    eos, list_tran_density = eos_creator.GetEOSType(**kwargs)


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
    pc14 = []
    dc14 = []
    mass = []
    radius = []
    lambda_ = []
    checkpoint_mass = []
    checkpoint_radius = []
    with wrapper.TidalLoveWrapper(eos) as tidal_love:
        pc_max, max_mass, _, _, _, _ = tidal_love.FindMaxMass()
        tidal_love.checkpoint = np.append(eos.GetPressure(np.array(list_tran_density), 0), [SurfacePressure])
        for tg in target_mass:
            try:
                pc14_tg, mass_tg, radius_tg, lambda_tg, checkpoint_mass_tg, checkpoint_radius_tg = tidal_love.FindMass(mass=tg, central_pressure0=150)
                if any(np.isnan([mass_tg, radius_tg, lambda_tg, pc14_tg])) or any(np.isnan(checkpoint_mass_tg)) or any(np.isnan(checkpoint_radius_tg)):
                    raise ValueError('Some of the calculated values are nan.')
                pc14.append(pc14_tg)
                mass.append(mass_tg)
                radius.append(radius_tg)
                lambda_.append(lambda_tg)
                checkpoint_mass.append(checkpoint_mass_tg)
                checkpoint_radius.append(checkpoint_radius_tg)


                # find the central density of 1.4 star
                try:
                    dc14.append(opt.newton(lambda x: eos.GetPressure(x, 0) - pc14_tg, x0=2*0.16))
                except RuntimeError as error:
                    dc14.append(0)
            except RuntimeError as error:
                raise ValueError('Failed to find %g solar mass properties for this EOS' % tg)
        if max_mass >= max_mass_req: 
            try:
                pc2, _, _, _, _, _ = tidal_love.FindMass(mass=max_mass_req, central_pressure0=300)
            except RuntimeError as error:
                raise ValueError('Failed to find %g solar mass properties for this EOS' % max_mass_req)
        else:
          pc2 = 0
        
    """
    Write results to dict and return
    """
    result = {'Model':name, 
              'PCentral2MOdot': pc2, 
              'PCentralMaxMass':pc_max, 
              'MaxMass': max_mass} 
    for tg, r, lamb, pc, cp_r, dc in zip(target_mass, radius, lambda_, pc14, checkpoint_radius, dc14):
        result['R(%g)'%tg] = r
        result['lambda(%g)'%tg] = lamb
        result['PCentral(%g)'%tg] = pc
        result['DensCentral(%g)'%tg] = dc
        for den, (index, cp_radius) in zip(list_tran_density, enumerate(cp_r)):
            result['RadiusCheckpoint%d(%g)' % (index, tg)] = cp_radius
            result['DensityCheckpoint%d(%g)' % (index, tg)] = den
    for key, val in kwargs.items():
        result[key] = val

    return result



def CalculatePolarizability(df, Output, comm, PBar=False, **kwargs):
    EOSType = kwargs['EOSType']
    #summary = SummarizeSkyrme(df, EOSType=EOSType)
    total = df.shape[0]

    """
    Tells ConsolePrinter which quantities to be printed in real time
    """
    title = ['Model', 'R(1.4)', 'lambda(1.4)', 'PCentral(1.4)']
    if PBar:
        printer = cp.ConsolePBar(title, comm=comm, total=total, **kwargs)
    else:
        printer = cp.ConsolePrinter(title, comm=comm, total=total)
    
    
    """
    Create multiple pools for parallel computation
    """
    name_list = [(index, row) for index, row in df.iterrows()]
    result = []
    #CalculateModel(name_list[0], **kwargs)
    with ProcessPool(max_workers=kwargs['nCPU']) as pool:
        future = pool.map(partial(CalculateModel, **kwargs), name_list, timeout=100)
        iterator = future.result()
        while True:
            try:
                new_result = next(iterator)
                result.append(new_result)
                printer.PrintContent(new_result)
            except StopIteration:
                break
            except ValueError as error:
                printer.PrintError(error)
            except TimeoutError as error:
                printer.PrintError(error)
            except ProcessExpired as error:
                printer.PrintError(error)
                #print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                printer.PrintError(error)
                #print("function raised %s" % error)
                #print(error.traceback)  # Python's traceback of remote process
            printer.ListenFor(0.1)

    printer.Close()            
    """
    Merge calculation data with Skyrme loaded data
    """
    if len(result) > 0:
        data = [val for val in result]
        data = pd.DataFrame.from_dict(data)
        data.set_index('Model', inplace=True)
      
        #cols_to_use = df.columns.difference(summary.columns)
        #data = pd.concat([df[cols_to_use], summary, data], axis=1, sort=True)
        data = pd.concat([df, data], axis=1)    
        data.index = df.index.map(str)
        #data.combine_first(summary)
        #data.combine_first(data)
        data.dropna(axis=0, how='any', inplace=True)
    else:
        data = None

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--Input", default="SkyrmeParameters/PawelSkyrme.csv", help="Name of the Skyrme input file (Default: SkyrmeParameters/PawelSkyrme.csv)")
    parser.add_argument("-o", "--Output", default="Result", help="Name of the CSV output (Default: Result)")
    parser.add_argument("-et", "--EOSType", default="EOS", help="Type of EOS. It can be: EOS, EOSNoPolyTrope, BESkyrme, OnlySkyrme (Default: EOS)")
    parser.add_argument("-sd", "--SkyrmeDensity", type=float, default=0.3, help="Density at which Skyrme takes over from crustal EOS (Default: 0.3)")
    parser.add_argument("-pp", "--PolyTropeDensity", type=float, default=3, help="Density at which Skyrme EOS ends. (Default: 3)")
    parser.add_argument("-td", "--TranDensity", type=float, default=0.001472, help="Density at which Crustal EOS ends (Default: 0.001472)")
    parser.add_argument("-pd", "--PRCTransDensity", type=float, default=None, help="Enable PRC automatic density transition. Value entered determine fraction of density that is represented by relativistic gas")
    parser.add_argument("-cs", "--CrustSmooth", type=float, default=0, help="degrees of smoothing. Reduce oscillation of speed of sound near crustal volumn")
    parser.add_argument("-mm", "--MaxMassRequested", type=float, default=2, help="Maximum Mass to be achieved for EOS in unit of solar mass (Default: 2)")
    parser.add_argument("-cf", "--CrustFileName", default='Constraints/EOSCrustOutput.dat', help="Type of crustal EoS used (Default: Constraints/EOSCrustOutput.dat)")
    parser.add_argument("-tg", "--TargetMass", type=float, nargs='+', default=[1.4], help="Target mass of the neutron star. (Default: 1.4)")
    args = parser.parse_args()

    df = LoadSkyrmeFile(args.Input)
    argd = vars(args)
    argd['TranDensity'] = argd['TranDensity']*rho0
    argd['SkyrmeDensity'] = argd['SkyrmeDensity']*rho0
    argd['PolyTropeDensity'] = argd['PolyTropeDensity']*rho0

    df = CalculatePolarizability(df, **argd)
    df = AddPressure(df)
    df.to_csv('Results/%s.csv' % args.Output, index=True)

