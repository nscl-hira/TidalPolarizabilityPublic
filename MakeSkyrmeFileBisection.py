from tqdm import tqdm
import shutil
import tempfile
import os
#!/projects/hira/tsangc/Polarizability/myPy/bin/python -W ignore
import sys
from multiprocessing import cpu_count
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
from collections import namedtuple

from Utilities.Utilities import FlattenListElements, ConcatenateListElements, DataIO
import Utilities.ConsolePrinter as cp
import TidalLove.TidalLoveWrapper as wrapper
from Utilities.Constants import *
from Utilities.MasterSlave import MasterSlave
from Utilities.EOSCreator import EOSCreator, SummarizeSkyrme
#from SelectPressure import AddPressure

p = configargparse.get_argument_parser()
if len(p._default_config_files) == 0:
    p._default_config_files.append('Default.ini')

p.add_argument('--PBar', dest='PBar', action='store_true', help="Enable if you don't need to display everything during calculation, just a progress bar")
p.add_argument('-tg', "--TargetMass", type=float, nargs='+', help="Target mass of the neutron star.")
p.add_argument("-mm", "--MaxMassRequested", type=float, help="Maximum Mass to be achieved for EOS in unit of solar mass")


OuterCrustDensity = 0.3e-3

def GenerateMetaDataFrame(filename='EOSComparsion.csv', size=100000, iter=0):
    df = pd.read_csv(filename)
    pars = list(df.columns)
    pars.remove('Name')
    pars.remove('Type')
    priors = []
    for model in set(df['Name']):
        if model == 'Total':
            continue
        Average = df[(df['Name'] == model) & (df['Type'] == 'Average')][pars].iloc[0].values
        Sigma = df[(df['Name'] == model) & (df['Type'] == 'Sigma')][pars].iloc[0].values

        values = np.random.uniform(Average - 4*Sigma, 
                                   Average + 4*Sigma, 
                                   size=(size, Average.shape[0])).T

        new_prior = {}
        for key, value in zip(pars, values):
            new_prior[key] = value.flatten()
        new_prior = pd.DataFrame(new_prior)
        new_prior['Model_Type'] = model
        priors.append(new_prior)

    priors = pd.concat(priors)
    priors.index = priors.index + iter*size
    priors.index = priors.index.map(str)
    return priors.fillna(0)



def LoadSkyrmeFile(filename):
    df = pd.read_csv(filename, index_col=0)
    df.index = df.index.map(str)
    return df.fillna(0)

def CheckCausality(eos, rho_max):
    try:
        rho = np.concatenate([np.logspace(np.log(1e-9), np.log(3.76e-4), 100, base=np.exp(1)), np.linspace(3.77e-4, rho_max, 900)])
        sound = np.array(eos.GetSpeedOfSound(rho, 0))
    except Exception as error:
        logger.exception('Causality cannot be determined')
        return {'ViolateCausality': True, 'NegSound': True, 'ViolateFrom': 0.}
    else:
        if all(sound <= 1) and all(sound >=0):
            return {'ViolateCausality': False, 'NegSound': False, 'ViolateFrom': 0.}
        elif any(sound > 1):
            idx = np.where(sound > 1)
            return {'ViolateCausality': True, 'NegSound': False, 'ViolateFrom': rho[idx][0]}
        else:
            idx = np.where(sound <= 0)
            return {'ViolateCausality': True, 'NegSound': True, 'ViolateFrom': rho[idx][0]}
 

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
    
"""
Print the selected EOS into a file for the tidallove script to run
"""
def CalculateModel(name_and_eos, EOSType, TargetMass, MaxMassRequested, Transform_kwargs):
    name = name_and_eos[0]
    Backbone_kwargs = name_and_eos[1]
    eos_creator = EOSCreator()

    """
    Prepare EOS
    """
    logger.debug('Preparing EOS %s', name)
    result = {}
    eos_check_result = {}
    meta_data = {}
    Transform_kwargs['MaxMass'] = MaxMassRequested
    try:
        eos, list_tran_density, new_kwargs = eos_creator.Factory(EOSType=EOSType, 
                                                                 Backbone_kwargs=Backbone_kwargs, 
                                                                 Transform_kwargs=Transform_kwargs)
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
            result['MaxMass'] = MaxMassResult

            if result['MaxMass'].mass >= MaxMassRequested: 
                logger.debug('Finding NS of required mass %s because maximum possible mass for EOS %s is larger than required' % 
                             (MaxMassRequested, name))
                TidalResult = tidal_love.FindMass(mass=MaxMassRequested)
            else:
                TidalResult = wrapper.TidalLoveResult(len(list_tran_density))
            result['Mass%g' % MaxMassRequested] = TidalResult

            for tg in TargetMass:
                logger.debug('Finding NS with mass %g for %s' % (tg, name))
                TidalResult = tidal_love.FindMass(mass=tg)
                result['Mass%g' % tg] = TidalResult

            if all(value.IsNan() for title, value in result.items()):
                 logger.debug('No NS can be formed with EOS %s' % name)
                 eos_check_result['NoData'] = True
            else:
                 eos_check_result['NoData'] = False

        logger.debug('Causality checking for EOS %s' % name)
        eos_check_result = {**eos_check_result, **CheckCausality(eos, result['MaxMass'].DensCentral)}

    # expand all results are dict
    for title, value in result.items():
        value.Checkpoint_dens = list_tran_density
    logger.debug('Creating summarize information for EOS %s' % name)
    summary = SummarizeSkyrme(eos_creator.nuclear_eos)
    logger.debug('Adding P, P_sym, S_sym information for EOS %s' % name)
    additional_info = AdditionalInfo(eos_creator.nuclear_eos)

    # wow that's a lot of things to unpack
    eos_info = namedtuple('EOSInfo', ['name', 'TOVresults', 'EOSDeriv', 'EOSValues', 'Meta', 'NewKwargs', 'BackboneKwargs', 'Causality'])
    return eos_info(name, result, summary, additional_info, meta_data, new_kwargs, Backbone_kwargs, eos_check_result)



def CalculatePolarizability(df, mslave, Output, EOSType, TargetMass, MaxMassRequested, **Transform_kwargs): 
    total = df.shape[0]

    """
    Tells ConsolePrinter which quantities to be printed in real time
    """
    name_list = [(index, row) for index, row in df.iterrows()]
    #CalculateModel(name_list[0], **kwargs)
    logger.debug('Begin multiprocess calculation')

    """
    Save meta data for every 10 EOSs
    """
    dataIO = DataIO('Results/%s.h5' % Output, flush_interval=1000)
    for new_result in tqdm(mslave.map(partial(CalculateModel, 
                                              EOSType=EOSType,
                                              TargetMass=TargetMass, 
                                              MaxMassRequested=MaxMassRequested,
                                              Transform_kwargs=Transform_kwargs),
                                       name_list,
                                       chunk_size=1000), 
                            total=total, 
                            ncols=100, 
                            smoothing=0.):
         try:
             name = new_result.name
             for title, result in new_result.TOVresults.items():
                 dataIO.AppendData('result', name, result.ToDict(), title)
             dataIO.AppendData('new_kwargs', name, new_result.NewKwargs)
             dataIO.AppendData('meta', name, new_result.Meta)
             dataIO.AppendData('kwargs', name, new_result.BackboneKwargs)
             dataIO.AppendData('summary', name, new_result.EOSDeriv)
             dataIO.AppendData('Additional_info', name, new_result.EOSValues)
             dataIO.AppendData('EOSCheck', name, new_result.Causality)
         except Exception:
             logger.exception('Cannot save data')
    try:
        dataIO.AppendMeta('kwargs', {'EOSType': EOSType, 'MaxMassRequested': MaxMassRequested, **Transform_kwargs})
        dataIO.Close()
    except Exception as error:
        logger.exception('Cannot close dataIO')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
logging.basicConfig(filename='log/app_rank%d.log' % rank, format='Process id %(process)d: %(name)s %(levelname)s - %(message)s', level=logging.DEBUG)
#logging.basicConfig(format='Process id %(process)d: %(name)s %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    mslave = MasterSlave(comm)
    p.add_argument("-i", "--Input", help="Name of the Skyrme input file")
    p.add_argument("-o", "--Output", help="Name of the CSV output (Default: Result)")
    p.add_argument("-et", "--EOSType", help="Type of EOS. It can be: EOS, EOSNoPolyTrope, BESkyrme, OnlySkyrme")
    p.add_argument('--Gen', dest='Gen', default=False, action='store_true', help="Enable if need to generate random parameters")
    p.add_argument("-s", "--Size", type=int, help="Size of the generated random parameters")
    p.add_argument('-it', "--Iter", type=int, help='Iterations of generated random parameters.')
    p.add_argument('--GenFile', help='Range of parameters for parameter generation.')


    args, unknown = p.parse_known_args()
    argd = vars(args)

    if args.Gen:
        argd['Output'] = argd['Output'] + '.Gen'
        for num_iter in range(args.Iter):
            logger.debug('Generating meta file')
            df = GenerateMetaDataFrame(args.GenFile, size=args.Size, iter=num_iter)
            logger.debug('Dataframe created')
            CalculatePolarizability(df, mslave, **argd)
    else:
        df = LoadSkyrmeFile(args.Input)
        CalculatePolarizability(df, mslave, **argd) 
    mslave.Close()

