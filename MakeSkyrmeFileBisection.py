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
import logging
from multiprocessing_logging import install_mp_handler, MultiProcessingHandler

import Utilities.ConsolePrinter as cp
import TidalLove.TidalLoveWrapper as wrapper
from Utilities.Constants import *
from Utilities.EOSCreator import EOSCreator, SummarizeSkyrme
from SelectPressure import AddPressure

OuterCrustDensity = 0.3e-3

logger = logging.getLogger(__name__)
install_mp_handler(logger)

def LoadSkyrmeFile(filename):
    df = pd.read_csv(filename, index_col=0)
    df.index = df.index.map(str)
    return df.fillna(0)

def CheckCausality(eos, rho_max):
    rho = np.concatenate([np.logspace(np.log(1e-9), np.log(3.76e-4), 100, base=np.exp(1)), np.linspace(3.77e-4, rho_max, 900)])
    sound = np.array(eos.GetSpeedOfSound(rho, 0))

    if all(sound <= 1) and all(sound >=0):
        return False, False
    elif any(sound <=0):
        return True, True
    else:
        return True, False

def AdditionalInfo(eos_creator):
    eos = eos_creator.ImportedEOS
    rho0 = eos.rho0
    return {'P(4rho0)':eos.GetPressure(4*rho0, 0),
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

def FindMaxMass(tidal_love):
    pcentral, max_mass, _, _, _, _ = tidal_love.FindMaxMass()
    return pcentral, max_mass

def FindAMass(tidal_love, eos, mass, cp_density_list):
    tidal_love.checkpoint = eos.GetPressure(np.array(cp_density_list), 0).tolist()
    result = tidal_love.FindMass(mass=mass, central_pressure0=150, tol=0.001, rtol=0.001)
    if any(np.isnan(result[:4])) or any(np.isnan(result[4:]).flatten()):
       logger.warning('Some of the calculated values are nan for mass %g.', mass)
       raise ValueError('Some of the calculated values are nan.')

    named_result = {'PCentral(%g)' % mass: result[0],
                    'R(%g)' % mass: result[2],
                    'lambda(%g)' % mass: result[3]}
    for den, (index, cp_radius) in zip(cp_density_list, enumerate(result[4])):
        named_result['RadiusCheckpoint%d(%g)' % (index, mass)] = cp_radius
        named_result['DensityCheckponit%d(%g)' % (index, mass)] = den

    # find the central density of 1.4 star
    try:
        named_result['DensCentral(%g)' % mass] = opt.newton(lambda x: eos.GetPressure(x, 0) - result[0], x0=2*0.16)
    except RuntimeError as error:
        logger.warning('Cannot find central density for mass %g' % mass)
        named_result['DensCentral(%g)' % mass] = 0

    return named_result
   


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
    
    logger.debug('Preparing EOS %s', name)
    kwargs = eos_creator.PrepareEOS(**kwargs)
    logger.debug('Getting EOS %s', name)
    eos, list_tran_density = eos_creator.GetEOSType(**kwargs)


    # insert surface density
    list_tran_density.append(OuterCrustDensity)

    """
    1.4 solar mass and 2.0 solar mass calculation
    """
    result = {'Model': str(name)}

    with wrapper.TidalLoveWrapper(eos) as tidal_love:
        logger.debug('Finding maximum mass for EOS %s', name)
        result['PCentralMaxMass'], result['MaxMass'] = FindMaxMass(tidal_love)
        for tg in target_mass:
            logger.debug('Finding NS with mass %g for %s' % (tg, name))
            result_each_mass = FindAMass(tidal_love, eos, tg, list_tran_density)
            result = {**result, **result_each_mass}
        if result['MaxMass'] >= max_mass_req: 
            logger.debug('Finding NS of required mass %s because maximum possible mass for EOS %s is larger than required' % (max_mass_req, name))
            result_max_mass_req = FindAMass(tidal_love, eos, max_mass_req, list_tran_density)
            result['PCentral2MOdot'] = result_max_mass_req['PCentral(%g)' % max_mass_req]
            result['MaxMassReq'] = max_mass_req

    logger.debug('Creating summarize information for EOS %s' % name)
    summary = SummarizeSkyrme(eos_creator)
    logger.debug('Adding P, P_sym, S_sym information for EOS %s' % name)
    additional_info = AdditionalInfo(eos_creator)
    logger.debug('Causality checking for EOS %s' % name)
    result['ViolateCausality'], result['NegSound'] = CheckCausality(eos, result['PCentralMaxMass'])

    result = {**result, **kwargs, **summary, **additional_info}
    return result



def CalculatePolarizability(df, Output, comm, PBar=False, **kwargs):
    EOSType = kwargs['EOSType']
    #summary = SummarizeSkyrme(df, EOSType=EOSType)
    total = df.shape[0]

    """
    Tells ConsolePrinter which quantities to be printed in real time
    """
    title = ['Model', 'R(1.4)', 'lambda(1.4)', 'PCentral(1.4)']
    logger.debug('Calling console printer')
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
    logger.debug('Begin multiprocess calculation')
    with ProcessPool(max_workers=kwargs['nCPU']) as pool:
        future = pool.map(partial(CalculateModel, **kwargs), name_list, timeout=100)
        iterator = future.result()
        while True:
            try:
                new_result = next(iterator)
                result.append(new_result)
                printer.PrintContent(new_result)
            except StopIteration:
                logger.debug('All calculations finished!')
                break
            except ValueError as error:
                printer.PrintError(error) 
                logger.exception('Value error')
            except TimeoutError as error:
                printer.PrintError(error)
                logger.exception('Timeout')
            except ProcessExpired as error:
                printer.PrintError(error)
                logger.exception('ProcessExpired')
            except Exception as error:
                logger.exception('General exception received')
                printer.PrintError(error)
            printer.ListenFor(0.1)

    printer.Close()            
    """
    Merge calculation data with Skyrme loaded data
    """
    if len(result) > 0:
        logger.debug('Results found. Merging.')

        data = [val for val in result]
        data = pd.DataFrame.from_dict(data)
        data['Model'] = data['Model'].astype(str)
        data.set_index('Model', inplace=True)
      
        cols_to_use = df.columns.difference(data.columns)
        data = pd.concat([df[cols_to_use], data], axis=1)    
        data.dropna(axis=0, how='any', inplace=True)
        
        data.index = data.index.map(str)
        logger.debug('merged')
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

