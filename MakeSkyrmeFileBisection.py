from tqdm import tqdm
import shutil
import tempfile
import os
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
from mpi4py import MPI

from Utilities.Utilities import FlattenListElements, ConcatenateListElements, DataIO
import Utilities.ConsolePrinter as cp
import TidalLove.TidalLoveWrapper as wrapper
from Utilities.Constants import *
from Utilities.MasterSlave import MasterSlave
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

def GenerateMetaDataFrame(filename='EOSComparison.csv'):
    df = pd.read_csv('EOSComparsion.csv')
    pars = ['Esat', 'Esym', 'Lsym', 'Ksat', 'Ksym', 'Qsat', 'Qsym', 'Zsat', 'Zsym', 'msat', 'kv']
    priors = []
    for model in set(df['Name']):
        if model == 'Total':
            continue
        Average = df[(df['Name'] == model) & (df['Type'] == 'Average')][pars].iloc[0]
        Sigma = df[(df['Name'] == model) & (df['Type'] == 'Sigma')][pars].iloc[0]

        Esat, Esym, Lsym, Ksat, Ksym, Qsat, Qsym, Zsat, Zsym, msat, kv = np.random.uniform(Average - 2*Sigma, Average + 2*Sigma, size=(100000, Average.shape[0])).T

        new_prior = pd.DataFrame({'Esat':Esat.flatten(), 
                                  'Esym':Esym.flatten(), 
                                  'Lsym':Lsym.flatten(), 
                                  'Ksat':Ksat.flatten(), 
                                  'Ksym':Ksym.flatten(), 
                                  'Qsat':Qsat.flatten(), 
                                  'Qsym':Qsym.flatten(), 
                                  'Zsat':Zsat.flatten(), 
                                  'Zsym':Zsym.flatten(), 
                                  'msat':msat.flatten(), 
                                  'kv':kv.flatten()})
        new_prior['Model_Type'] = model
        priors.append(new_prior)

    priors = pd.concat(priors)
    priors.index = priors.index.map(str)
    return priors.fillna(0)



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
    

def RenameResultKey(tidal_results, mass=None):
    if mass is None:
        mass = tidal_results['mass']
    RenamedResults = {'PCentral(%g)'%mass : tidal_results['PCentral'],
                      'DensCentral(%g)'%mass : tidal_results['DensCentral'],
                      'R(%g)'%mass : tidal_results['Radius'],
                      'lambda(%g)'%mass : tidal_results['Lambda']}

    for index, (cp_mass, cp_radius) in enumerate(zip(tidal_results['Checkpoint_mass'], 
                                                              tidal_results['Checkpoint_radius'])):
        RenamedResults['RadiusCheckpoint%d(%g)' % (index, mass)] = cp_radius
        RenamedResults['MassCheckpoint%d(%g)' % (index, mass)] = cp_mass
        #RenamedResults['DensityCheckpoint%d(%g)' % (index, mass)] = cp_dens
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
    meta_data = {}
    try:
        eos, list_tran_density, kwargs = eos_creator.PrepareEOS(**{'EOSType': EOSType, 
                                                                   'MaxMassRequested': MaxMassRequested,
                                                                   **name_and_eos[1], 
                                                                   **kwargs})
        meta_data = eos_creator.GetMetaData()
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
            MaxMassResult = tidal_love.FindMaxMass()
            result['PCentralMaxMass'] = MaxMassResult['PCentral']
            result['MaxMass']  = MaxMassResult['mass']
            result['DensCentralMax'] = MaxMassResult['DensCentral']

            if result['MaxMass'] >= MaxMassRequested: 
                logger.debug('Finding NS of required mass %s because maximum possible mass for EOS %s is larger than required' % (MaxMassRequested, name))
                TidalResult = RenameResultKey(tidal_love.FindMass(mass=MaxMassRequested), MaxMassRequested)
            else:
                TidalResult = RenameResultKey({'PCentral': np.nan, 
                                               'DensCentral': np.nan, 
                                               'Radius': np.nan, 
                                               'Lambda': np.nan, 
                                               'Checkpoint_mass': [np.nan]*len(list_tran_density), 
                                               'Checkpoint_radius': [np.nan]*len(list_tran_density)}, MaxMassRequested)
            result = {**result, **TidalResult}

            for tg in TargetMass:
                logger.debug('Finding NS with mass %g for %s' % (tg, name))
                TidalResult = RenameResultKey(tidal_love.FindMass(mass=tg), tg)
                result = {**result, **TidalResult}

            if all(np.isnan(value) for value in result.values()):
                 logger.debug('No NS can be formed with EOS %s' % name)
                 result['NoData'] = True
            else:
                 result['NoData'] = False


        logger.debug('Causality checking for EOS %s' % name)
        try:
            result['ViolateCausality'], result['NegSound'], result['ViolateFrom'] = CheckCausality(eos, result['DensCentralMax'])
            #result['dUrca'] = dUrca(eos_creator, result['DensCentral(1.4)'])
        except Exception as error:
            logger.exception('Causality cannot be determined')
            result['ViolateCausality'], result['NegSound'], result['ViolateFrom'] = True, True, 0
 
    for index, cp_dens in enumerate(list_tran_density):
        result['DensityCheckpoint%d' % index] = cp_dens

    #result['Model'] = name
    logger.debug('Creating summarize information for EOS %s' % name)
    summary = SummarizeSkyrme(eos_creator.ImportedEOS)
    logger.debug('Adding P, P_sym, S_sym information for EOS %s' % name)
    additional_info = AdditionalInfo(eos_creator.ImportedEOS)

    return name, kwargs, result, summary, additional_info, meta_data
    #return {**kwargs, **result, **summary, **additional_info}, meta_data



def CalculatePolarizability(df, mslave, Output, **kwargs): 
    total = df.shape[0]
    args, unknown = p.parse_known_args()
    kwargs = {**kwargs, **vars(args)}

    """
    Tells ConsolePrinter which quantities to be printed in real time
    """
    name_list = [(index, row) for index, row in df.iterrows()]
    #CalculateModel(name_list[0], **kwargs)
    logger.debug('Begin multiprocess calculation')

    """
    Save meta data for every 10 EOSs
    """
    dataIO = DataIO('Results/%s.h5' % Output, flush_interval=500)
    for new_result in tqdm(mslave.map(partial(CalculateModel, **kwargs), name_list, chunk_size=100), total=total, ncols=100, smoothing=0.):
         try:
             name = new_result[0]
             dataIO.AppendData('meta', name, new_result[5])
             dataIO.AppendData('kwargs', name, new_result[1])
             dataIO.AppendData('result', name, new_result[2])
             dataIO.AppendData('summary', name, new_result[3])
             dataIO.AppendData('Additional_info', name, new_result[4])
         except Exception:
             logger.exception('Cannot save meta')
    try:
        dataIO.Close()
    except Exception as error:
        logger.exception('Cannot close dataIO')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
logging.basicConfig(filename='log/app_rank%d.log' % rank, format='Process id %(process)d: %(name)s %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    mslave = MasterSlave(comm)
    p.add_argument("-i", "--Input", help="Name of the Skyrme input file")
    p.add_argument("-o", "--Output", help="Name of the CSV output (Default: Result)")
    p.add_argument("-et", "--EOSType", help="Type of EOS. It can be: EOS, EOSNoPolyTrope, BESkyrme, OnlySkyrme")
    p.add_argument('--Gen', dest='Gen', action='store_true', help="Enable if need to generate random parameters")


    args, unknown = p.parse_known_args()
    argd = vars(args)

    num_iter = 0
    output = argd['Output']
    if args.Gen:
        while True:
            logger.debug('Generating meta file')
            df = GenerateMetaDataFrame()
            logger.debug('Dataframe created')
            argd['Output'] = '%s_%d' % (output, num_iter)
            CalculatePolarizability(df, mslave, **argd)
            num_iter += 1
    else:
        df = LoadSkyrmeFile(args.Input)
        CalculatePolarizability(df, mslave, **argd) 
    mslave.Close()

