#from pebble import ProcessPool
import configargparse
from autograd import elementwise_grad as egrad
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import scipy.optimize as optimize
import pandas as pd
import scipy.optimize as opt
from scipy.interpolate import UnivariateSpline
import sys
import logging
from multiprocessing_logging import install_mp_handler, MultiProcessingHandler

from TidalLove import TidalLoveWrapper as wrapper
import Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from Utilities.BetaEquilibrium import BetaEquilibrium

logger = logging.getLogger(__name__)
install_mp_handler(logger)

p = configargparse.get_argument_parser()
if len(p._default_config_files) == 0:
    p._default_config_files.append('Default.ini')

p.add_argument("-pd", "--PRCTransDensity", type=float,  help="Enable PRC automatic density transition. Value entered determine fraction of density that is represented by relativistic gas")
p.add_argument("-cf", "--CrustFileName", help="Type of crustal EoS used")
p.add_argument("-pp", "--PolyTropeDensity", type=float, help="Density at which Skyrme EOS ends.")
p.add_argument("-cs", "--CrustSmooth", type=float, help="degrees of smoothing. Reduce oscillation of speed of sound near crustal volumn")
p.add_argument("-sp", "--SpeedOfSound", type=float, help="Speed of sound at transition density. Required for MetaSound")


def FindCrustalTransDensity(Skryme):
    """
    This function uses result from PhysRevC.83.048510
    which contains a formula calculated empirically
    where density transition occurs
    Please refer to equation 17 - 24 in the paper for details
    """
    return -3.75e-4*Skryme.GetL(Skryme.rho0) + 0.0963 # fm-3

def FindDensityWhenSoundEquals(eos, speed):
    try:
       density = opt.newton(lambda x: eos.GetSpeedOfSound(x) - speed, x0=2*0.16)
    except Exception as error:
       logger.exception('Cannot find density where speed of sound = %g c' % speed)
       density = np.nan
    return density

def NuclearEOSFactory(EOSType, kwargs):
    if EOSType == 'Rod':
        # load Rodrigo EFT functional
        df_E = pd.read_csv('SkyrmeParameters/Rodrigo_extended.csv')
        df_Sym = pd.read_csv('SkyrmeParameters/Rodrigo_sym_extended.csv')
        eos = sky.SplineEOS.Construct(df_E['rho(fm-3)'], energy=df_E[kwargs['Name']] + 931.8, 
                                      rho_Sym=df_Sym['rho(fm-3)'], Sym=df_Sym[kwargs['Name']])
    elif EOSType == 'Power' or EOSType == 'PowerNoPolyTrope':
        eos = sky.PowerLawEOS(kwargs)
    elif EOSType == 'Meta' or EOSType == 'Meta2Poly' or EOSType == 'MetaSound':
        eos = sky.MetaModeling(kwargs)
    else: # default Nuclear EOS is Skyrme
        eos = sky.Skryme(kwargs)
    return eos

def AdjustPoly(eos, MaxMass):
    def FixMaxMass(pressure):
        eos.eos_list[-1].ChangeFinalPressure(7*0.16, pressure)
        with wrapper.TidalLoveWrapper(eos) as tidal_love:
            result = tidal_love.FindMaxMass()
            pc = result['PCentral']
            max_m = result['mass']
            if np.isnan(max_m): 
                raise ValueError('Mass maximization failed')
        return max_m - MaxMass
 
    PressureHigh = opt.newton(FixMaxMass, x0=500, rtol=0.0001, tol=0.0001)
    return PressureHigh

class EOSCreator:

    def __init__(self):
        self.density_list = []
        self.eos_list = []       
        self.rho = []
        self.pfrac = []
        self.mufrac = []
        self.energy = []

    def AddBackboneEOS(self, NuclearEOS, need_be=True, meta=None):
        self.nuclear_eos = NuclearEOS
        if need_be:
            if meta is None:
                be = BetaEquilibrium(self.nuclear_eos, np.linspace(0.01, 10., 100))
                self.backboneEOS = be[0]
                self.rho = be[1]
                self.pfrac = be[2]
                self.mufrac = be[3]
                self.energy = be[4]
            else:
                self.rho = meta_data['rho']
                self.pfrac = meta_data['pfrac']
                self.mufrac = meta_data['mufrac']
                self.energy = meta_data['energy']
                self.backboneEOS = sky.SplineEOS.Construct(self.rho, self.energy)
        else:
            self.backboneEOS = self.nuclear_eos

    def GetMetaData(self):
        return {'rho': self.rho, 'pfrac': self.pfrac, 'mufrac': self.mufrac, 'energy': self.energy}



    def Factory(self, EOSType, Backbone_kwargs, Transform_kwargs, meta=None):
        # combine default Transform_kwargs with default Transform_kwargs
        args, unknown = p.parse_known_args()
        Transform_kwargs = {**vars(args), **Transform_kwargs}
        
        # nuclear eos
        eos = NuclearEOSFactory(EOSType, Backbone_kwargs)
        need_be = False if EOSType == 'EOSNoCrust' else True
        
        #add this to this class
        self.AddBackboneEOS(eos, need_be, meta=meta)
        # get to work!
        if EOSType == 'EOS' or EOSType == '3Poly' or EOSType == 'Rod' or EOSType == 'Power':
            eos, new_kwargs  = self.BuildPoly(**Transform_kwargs)
        elif EOSType == 'EOS2Poly' or EOSType == 'Meta2Poly' or EOSType == 'MetaSound': 
            if EOSType == 'MetaSound':
                Transform_kwargs['PolyTropeDensity'] = None
            eos, new_kwargs = self.BuildSound(**Transform_kwargs)
        elif EOSType == 'EOSNoPolyTrope' or EOSType == 'PowerNoPolyTrope' or EOSType == 'Meta':
            eos, new_kwargs = self.BuildNoPolyTrope(**Transform_kwargs)
        elif EOSType == 'EOSNoCrust':
            eos, new_kwargs = self.BuildNoCrust(**Transform_kwargs)
        return eos, [rho[0] for rho in self.density_list[1:]], new_kwargs
        

    def BuildSound(self, CrustFileName, CrustSmooth, PRCTransDensity, SpeedOfSound, PolyTropeDensity=None, **kwargs):
        # find transition densities
        SkyrmeDensity = FindCrustalTransDensity(self.nuclear_eos)
        TranDensity = PRCTransDensity*SkyrmeDensity
        if PolyTropeDensity is None:
            PolyTropeDensity = FindDensityWhenSoundEquals(self.backboneEOS, SpeedOfSound)
            if np.isnan(PolyTropeDensity):
                PolyTropeDensity = 10.

        #construct EOS
        self.InsertEOS(self.ConstructCrust(CrustFileName, CrustSmooth), TranDensity) 
        self.InsertSmoothConnection(SkyrmeDensity)
        self.InsertEOS(self.backboneEOS, PolyTropeDensity)
        self.InsertEOS(lambda prev_density, prev_eos, next_density, next_eos:
                       sky.ConstSpeed.MatchBothEnds(prev_density,
                                                    prev_eos,
                                                    next_density,
                                                    next_eos), 10*0.16)
        return self.Build(), {'PolyTropeDensity': PolyTropeDensity}

    def BuildPoly(self, CrustFileName, CrustSmooth, PRCTransDensity, PolyTropeDensity, MaxMass, PressureHigh=None, **kwargs):
        # find transition densities
        SkyrmeDensity = FindCrustalTransDensity(self.nuclear_eos)
        TranDensity = PRCTransDensity*SkyrmeDensity

        #construct EOS
        self.InsertEOS(self.ConstructCrust(CrustFileName, CrustSmooth), TranDensity) 
        self.InsertConnection(SkyrmeDensity)
        self.InsertEOS(self.backboneEOS, PolyTropeDensity)
        if PressureHigh is None:
            PressureHigh = 500
            self.InsertEOS(lambda prev_density, prev_eos, next_density, next_eos:
                           sky.PolyTrope(prev_density, prev_eos.GetEnergy(prev_density),
                                         prev_eos.GetPressure(prev_density), 7*0.16, PressureHigh), 100)
            eos = self.Build()
            AdjustPoly(eos, MaxMass)
        else:
            self.InsertEOS(lambda prev_density, prev_eos, next_density, next_eos:
                           sky.PolyTrope(prev_density, prev_eos.GetEnergy(prev_density), 
                                         prev_eos.GetPressure(prev_density), 7*0.16, PressureHigh), 100)
            eos = self.Build()
        return eos, {'PressureHigh', PressureHigh}

    def BuildNoPolyTrope(self, CrustFileName, CrustSmooth, PRCTransDensity, **kwargs):
        # find transition densities
        SkyrmeDensity = FindCrustalTransDensity(self.nuclear_eos)
        TranDensity = PRCTransDensity*SkyrmeDensity

        #construct EOS
        self.InsertEOS(self.ConstructCrust(CrustFileName, CrustSmooth), TranDensity) 
        self.InsertSmoothConnection(SkyrmeDensity)
        self.InsertEOS(self.backboneEOS, 100)

        return self.Build(), None

    def BuildNoCrust(self, **kwargs):
        self.InsertEOS(self.backboneEOS, 100)
        return self.Build(), None


    def InsertConnection(self, end_density):
        self.InsertEOS(lambda prev_density, prev_eos, next_density, next_eos:
                              sky.PseudoEOS.MatchBothEnds(prev_density,
                                                         prev_eos,
                                                         next_density, 
                                                         next_eos), end_density)
    def InsertSmoothConnection(self, end_density):
        self.InsertEOS(lambda prev_density, prev_eos, next_density, next_eos:
                       sky.SplineEOS.SmoothConnection(prev_density, 
                                                      prev_eos, 
                                                      sky.PseudoEOS, 
                                                      next_density, 
                                                      next_eos), end_density)



    def ConstructCrust(self, CrustFilename, CrustSmooth):
        crust = pd.read_csv(CrustFilename)
        return sky.SplineEOS.Construct(crust['rho(fm-3)'].values,
                                       energy_density=crust['E(MeV/fm3)'].values,
                                       smooth=CrustSmooth,
                                       pressure=crust['P(MeV/fm3)'].values)

    def InsertEOS(self, eos, next_density):
        if len(self.density_list) == 0:
            self.density_list.append((0, next_density))
        else:
            self.density_list.append((self.density_list[-1][1], next_density))
        self.eos_list.append(eos)

    def Build(self):
        # build remaining EOSs
        num_eos = len(self.eos_list)
        for idx, (density_interval, eos) in enumerate(zip(self.density_list, self.eos_list)):
            if hasattr(eos, '__call__'):
                if idx == 0:
                    prev_eos = None
                else:
                    prev_eos = self.eos_list[idx-1]
                if idx >= num_eos-1:
                    next_eos = None
                else:
                    next_eos = self.eos_list[idx+1]
                self.eos_list[idx] = eos(density_interval[0], prev_eos,
                                         density_interval[1], next_eos)
        return sky.EOSConnect(self.density_list, self.eos_list)
 
def SummarizeSkyrme(sky_eos):
    """
    This function will print out the value of E0, K0, K'=-Q0, J=S(rho0), L, Ksym, Qsym, m*
    """

    try:
        rho0 = sky_eos.rho0
        E0 = sky_eos.GetEnergy(rho0, 0.5)
        K0 = sky_eos.GetK(rho0, 0.5)
        Kprime = -sky_eos.GetQ(rho0, 0.5)
        Z0 = sky_eos.GetZ(rho0, 0.5)
        J = sky_eos.GetAsymEnergy(rho0)
        L = sky_eos.GetL(rho0)
        Ksym = sky_eos.GetKsym(rho0)
        Qsym = sky_eos.GetQsym(rho0)
        Zsym = sky_eos.GetZsym(rho0)
        summary_dict = {'E0':E0, 'K0':K0, 'K\'':Kprime, 'Z0': Z0, 'J':J, 'L':L, 'Ksym':Ksym, 'Qsym':Qsym, 'Zsym':Zsym}
    except Exception:
        raise Exception('The EOS type does not suppor calculation of L, K and Q.')
    try:
        eff_m = sky_eos.GetEffectiveMass(rho0, 0.5)
        m_s = sky_eos.GetMs(rho0)
        m_v = sky_eos.GetMv(rho0)
        fi = sky_eos.GetFI(rho0)
        summary_dict['m*'] = eff_m
        summary_dict['m_s'] = m_s
        summary_dict['m_v'] = m_v
        summary_dict['fi'] = fi
    except Exception:
        pass
 
    return summary_dict



if __name__ == "__main__":
    df = pd.read_csv('SkyrmeParameters/PawelSkyrme.csv', index_col=0)
    df.fillna(0, inplace=True)
 
    creator = EOSCreator()
    print(df.iloc[117])
    eos, density_list, _ = creator.Factory(EOSType='EOS', Backbone_kwargs=df.iloc[117],
                                           Transform_kwargs={'PRCTransDensity': 0.3, 
                                                             'PolyTropeDensity': 3*0.16, 
                                                             'MaxMass': 2})
    print(density_list)
    nuclear_eos = NuclearEOSFactory('EOSNoPolyTrope', df.iloc[117])
    density_list = [1e-9] + density_list + [10]
    print(density_list)
    for low, up, color in zip(density_list[:-1], density_list[1:], ['yellow', 'green', 'blue', 'red']):
        density = np.logspace(np.log(low), np.log(up), 100, base=np.e)
        pressure = eos.GetPressure(density)
        energy = eos.GetEnergyDensity(density)
        plt.plot(energy, pressure, color=color)
    density = np.linspace(1e-4, 10, 1000)
    plt.plot(nuclear_eos.GetEnergyDensity(density), nuclear_eos.GetPressure(density), color='black')
    plt.xlim([1e-2, 1e4])
    plt.ylim([1e-4, 1e4])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
