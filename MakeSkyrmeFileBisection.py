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
import configargparse   

import Utilities.ConsolePrinter as cp
import TidalLove.TidalLoveWrapper as wrapper
from Utilities.Constants import *
from Utilities.EOSCreator import EOSCreator, SummarizeSkyrme
from SelectPressure import AddPressure

p = configargparse.get_argument_parser()
if len(p._default_config_files) == 0:
    p._default_config_files.append('Default.ini')

p.add_argument('--PBar', dest='PBar', action='store_true', help="Enable if you don't need to display everything during calculation, just a progress bar")
p.add_argument('-c', "--nCPU", type=int, help="Number of CPU used in each nodes")
p.add_argument('-tg', "--TargetMass", type=float, nargs='+', help="Target mass of the neutron star.")
p.add_argument("-mm", "--MaxMassRequested", type=float, help="Maximum Mass to be achieved for EOS in unit of solar mass")


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
        return False, False, 0
    elif any(sound > 1):
        idx = np.where(sound > 1)
        return True, False, rho[idx][0]
    else:
        idx = np.where(sound <= 0)
        return True, True, rho[idx][0]
 

def AdditionalInfo(eos):
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

def dUrca(eos_creator, density):
    xep = eos_creator.pfrac*(1-eos_creator.mufrac)/eos_creator.pfrac
    xDU = 1./(1.+(1.+np.power(xep, 1/3.)**3))
    idx = np.abs(eos_creator.rho - density).argmin()
    return any(eos_creator.pfrac[:idx] > xDU[:idx])
    

def RenameResultKey(tidal_results, cp_checkpoints, mass=None):
    if mass is None:
        mass = tidal_results['mass']
    RenamedResults = {'PCentral(%g)'%mass : tidal_results['PCentral'],
                      'DensCentral(%g)'%mass : tidal_results['DensCentral'],
                      'R(%g)'%mass : tidal_results['Radius'],
                      'lambda(%g)'%mass : tidal_results['Lambda']}

    for index, (cp_dens, cp_mass, cp_radius) in enumerate(zip(cp_checkpoints, 
                                                              tidal_results['Checkpoint_mass'], 
                                                              tidal_results['Checkpoint_radius'])):
        RenamedResults['RadiusCheckpoint%d(%g)' % (index, mass)] = cp_radius
        RenamedResults['MassCheckpoint%d(%g)' % (index, mass)] = cp_mass
        RenamedResults['DensityCheckpoint%d(%g)' % (index, mass)] = cp_dens
    return RenamedResults


"""
Print the selected EOS into a file for the tidallove script to run
"""
def CalculateModel(name_and_eos, EOSType, MaxMassRequested, TargetMass, **kwargs):
    name = str(name_and_eos[0])
    eos_creator = EOSCreator()

    """
    Prepare EOS
    """
    logger.debug('Preparing EOS %s', name)
    result = {}
    try:
        eos, list_tran_density, kwargs = eos_creator.PrepareEOS(**{'EOSType': EOSType, 
                                                                   'MaxMassRequested': MaxMassRequested,
                                                                   **name_and_eos[1], 
                                                                   **kwargs})
    except Exception:
        logger.exception('EOS cannot be created')
    else:
        # insert surface density
        list_tran_density.append(OuterCrustDensity)
        """
        1.4 solar mass and 2.0 solar mass calculation
        """
    
        with wrapper.TidalLoveWrapper(eos) as tidal_love:
            tidal_love.density_checkpoint = list_tran_density
            logger.debug('Finding maximum mass for EOS %s', name)
            try:
                MaxMassResult = tidal_love.FindMaxMass()
                result['PCentralMaxMass'] = MaxMassResult['PCentral']
                result['MaxMass']  = MaxMassResult['mass']
                result['DensCentralMax'] = MaxMassResult['DensCentral']

                if result['MaxMass'] >= MaxMassRequested: 
                    logger.debug('Finding NS of required mass %s because maximum possible mass for EOS %s is larger than required' % (MaxMassRequested, name))
                    TidalResult = RenameResultKey(tidal_love.FindMass(mass=MaxMassRequested), list_tran_density, MaxMassRequested)
                    result = {**result, **TidalResult}
            except:
                logger.warning('Cannot find maximum mass for %s' % name)

            for tg in TargetMass:
                try:
                    logger.debug('Finding NS with mass %g for %s' % (tg, name))
                    TidalResult = RenameResultKey(tidal_love.FindMass(mass=tg), list_tran_density, tg)
                    result = {**result, **TidalResult}
                except:
                    logger.warning('Cannot form NS with mass %g for %s' % (tg, name))
        logger.debug('Causality checking for EOS %s' % name)
        try:
            result['ViolateCausality'], result['NegSound'], result['ViolateFrom'] = CheckCausality(eos, result['DensCentralMax'])
            #result['dUrca'] = dUrca(eos_creator, result['DensCentral(1.4)'])
        except Exception as error:
            logger.exception('Causality cannot be determined')
 
                
    if not bool(result):
        logger.debug('No NS can be formed with EOS %s' % name)
        result['NoData'] = True
    else:
        result['NoData'] = False

    result['Model'] = name
    logger.debug('Creating summarize information for EOS %s' % name)
    summary = SummarizeSkyrme(eos_creator.ImportedEOS)
    logger.debug('Adding P, P_sym, S_sym information for EOS %s' % name)
    additional_info = AdditionalInfo(eos_creator.ImportedEOS)

    return {**kwargs, **result, **summary, **additional_info}



def CalculatePolarizability(df, Output, comm, **kwargs): 
    total = df.shape[0]
    args, unknown = p.parse_known_args()
    kwargs = {**kwargs, **vars(args)}

    """
    Tells ConsolePrinter which quantities to be printed in real time
    """
    title = ['Model', 'R(1.4)', 'lambda(1.4)', 'PCentral(1.4)']
    logger.debug('Calling console printer')
    if kwargs['PBar']:
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
        data.dropna(axis=0, how='all', inplace=True)
        data = pd.concat([df[cols_to_use].loc[data.index], data], axis=1)    

        logger.debug('merged')
        return data
    else:
        return None

if __name__ == "__main__":
    p.add_argument("-i", "--Input", help="Name of the Skyrme input file")
    p.add_argument("-o", "--Output", help="Name of the CSV output (Default: Result)")
    p.add_argument("-et", "--EOSType", help="Type of EOS. It can be: EOS, EOSNoPolyTrope, BESkyrme, OnlySkyrme")
    args, unknown = p.parse_known_args()
    argd = vars(args)
    print(argd)
    """
    df = LoadSkyrmeFile(args.Input)

    df = CalculatePolarizability(df, **argd)
    """
