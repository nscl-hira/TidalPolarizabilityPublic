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
    df.index = df.index.map(str)
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
    result = {'Model': str(name)}

    with wrapper.TidalLoveWrapper(eos) as tidal_love:
        result['PCentralMaxMass'], result['MaxMass'], _, _, _, _ = tidal_love.FindMaxMass()
        tidal_love.checkpoint = np.append(eos.GetPressure(np.array(list_tran_density), 0), [SurfacePressure])
        for tg in target_mass:
            try:
                tg_result = tidal_love.FindMass(mass=tg, central_pressure0=150)
                if any(np.isnan(tg_result[:4])) or any(np.isnan(tg_result[4:]).flatten()):
                    raise ValueError('Some of the calculated values are nan.')

                result['PCentral(%g)' % tg] = tg_result[0]
                result['R(%g)' % tg] = tg_result[2]
                result['lambda(%g)' % tg] = tg_result[3]
                for den, (index, cp_radius) in zip(list_tran_density, enumerate(tg_result[4])):
                    result['RadiusCheckpoint%d(%g)' % (index, tg)] = cp_radius
                    result['DensityCheckpoint%d(%g)' % (index, tg)] = den

                # find the central density of 1.4 star
                try:
                    result['DensCentral(%g)' % tg] = opt.newton(lambda x: eos.GetPressure(x, 0) - tg_result[0], x0=2*0.16)
                except RuntimeError as error:
                    result['DensCentral(%g)' % tg] = 0
            except RuntimeError as error:
                raise ValueError('Failed to find %g solar mass properties for this EOS' % tg)
        if result['MaxMass'] >= max_mass_req: 
            try:
                result['PCentral2MOdot'], result['MaxMassReq'], _, _, _, _ = tidal_love.FindMass(mass=max_mass_req, central_pressure0=300)
            except RuntimeError as error:
                raise ValueError('Failed to find %g solar mass properties for this EOS' % max_mass_req)
        else:
          result['PCentral2MOdot'] = 0
    summary = SummarizeSkyrme(eos_creator)
    eos = eos_creator.ImportedEOS
    pressure = {'P(4rho0)':eos.GetPressure(4*rho0, 0),
                'P(3.5rho0)':eos.GetPressure(3.5*rho0, 0),
                'P(3rho0)':eos.GetPressure(3*rho0, 0),
                'P(2rho0)':eos.GetPressure(2*rho0, 0),
                'P(1.5rho0)':eos.GetPressure(1.5*rho0, 0),
                'P(rho0)':eos.GetPressure(rho0, 0),
                'P(0.67rho0)':eos.GetPressure(0.67*rho0, 0),
                'P_Sym(4rho0)':eos.GetPressure(4*rho0, 0.5),
                'P_Sym(3.5rho0)':eos.GetPressure(3.5*rho0, 0.5),
                'P_Sym(3rho0)':eos.GetPressure(3*rho0, 0.5),
                'P_Sym(2rho0)':eos.GetPressure(2*rho0, 0.5),
                'P_Sym(1.5rho0)':eos.GetPressure(1.5*rho0, 0.5),
                'P_Sym(rho0)':eos.GetPressure(rho0, 0.5),
                'P_Sym(0.67rho0)':eos.GetPressure(0.67*rho0, 0.5),
                'Sym(4rho0)':eos.GetAsymEnergy(4*rho0),
                'Sym(3.5rho0)':eos.GetAsymEnergy(3.5*rho0),
                'Sym(3rho0)':eos.GetAsymEnergy(3*rho0),
                'Sym(2rho0)':eos.GetAsymEnergy(2*rho0),
                'Sym(1.5rho0)':eos.GetAsymEnergy(1.5*rho0),
                'Sym(rho0)':eos.GetAsymEnergy(rho0),
                'Sym(0.67rho0)':eos.GetAsymEnergy(0.67*rho0),
                'L(2rho0)':eos.GetL(2*rho0),
                'L(1.5rho0)':eos.GetL(1.5*rho0),
                'L(rho0)':eos.GetL(rho0),
                'L(0.67rho0)':eos.GetL(0.67*rho0)}

    result = {**result, **kwargs, **summary, **pressure}
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
        data['Model'] = data['Model'].astype(str)
        data.set_index('Model', inplace=True)

      
        cols_to_use = df.columns.difference(data.columns)
        #data = pd.concat([df[cols_to_use], summary, data], axis=1, sort=True)
        data = pd.concat([df[cols_to_use], data], axis=1)    
        data.dropna(axis=0, how='any', inplace=True)
        
        data.index = data.index.map(str)
        #

        #data.combine_first(summary)
        #data.combine_first(data)

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

