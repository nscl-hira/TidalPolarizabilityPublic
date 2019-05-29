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
p.add_argument("-td", "--TranDensity", type=float, help="Transition density where crustal EOS stops. Ignored if PRCTransDensity > 0")
p.add_argument("-sd", "--SkyrmeDensity", type=float, help="Density at which Skyrme/Meta/PowerLaw/Rod EOS begins. Ignored if PRCTransDensity > 0")

def FindCrustalTransDensity(Skryme):
    """
    This function uses result from PhysRevC.83.048510
    which contains a formula calculated empirically
    where density transition occurs
    Please refer to equation 17 - 24 in the paper for details
    """
    
    return -3.75e-4*Skryme.GetL(Skryme.rho0) + 0.0963 # fm-3

class EOSCreator:
    placeholder4connection = 13256

    def __init__(self):
        self.EQType = None # type of EOS for which equilibrium is calculated
        self.BENuclear = None 
        self.rho = None 
        self.pfrac = None 
        self.mufrac = None
        self.energy = None
        self.ImportedEOS = None
        self.crustEOS = None

    def ImportEOS(self, **kwargs):
        """
        Only load the externally imported EOS
        No beta equilibrium for fast EOS fetching
        """
        args, unknown = p.parse_known_args()
        kwargs = {**vars(args), **kwargs}
        logger.debug('Combined arguments: ' + str(kwargs))
        EOSType = kwargs['EOSType']
        if EOSType == 'Rod':
            # load Rodrigo EFT functional
            df_E = pd.read_csv('SkyrmeParameters/Rodrigo_extended.csv')
            df_Sym = pd.read_csv('SkyrmeParameters/Rodrigo_sym_extended.csv')
            self.ImportedEOS = sky.SplineEOS.Construct(df_E['rho(fm-3)'], energy=df_E[kwargs['Name']] + 931.8, 
                                                       rho_Sym=df_Sym['rho(fm-3)'], Sym=df_Sym[kwargs['Name']])
        elif EOSType == 'Power' or EOSType == 'PowerNoPolyTrope':
            self.ImportedEOS = sky.PowerLawEOS(self.row)
        elif EOSType == 'Meta' or EOSType == 'Meta2Poly':
            if 'msat' not in kwargs:
                kwargs['msat'] = 0.73
                kwargs['kv'] = 0.46
            self.ImportedEOS = sky.MetaModeling(kwargs)
        else:
            self.ImportedEOS = sky.Skryme(kwargs)

    def PrepareEOS(self, meta_data=None, **kwargs):
        args, unknown = p.parse_known_args()
        kwargs = {**vars(args), **kwargs}
        EOSType = kwargs['EOSType']
        self.ImportEOS(**kwargs)
       
        if kwargs['PRCTransDensity'] > 0:
            kwargs['TranDensity'] = kwargs['PRCTransDensity']*FindCrustalTransDensity(self.ImportedEOS)
            kwargs['SkyrmeDensity'] = FindCrustalTransDensity(self.ImportedEOS)

        if EOSType != 'EOSNoCrust':
            if meta_data is None:
                if EOSType == 'EOSNoPolyTrope' or EOSType == 'PowerNoPolyTrope' or EOSType == 'Meta':
                    logger.debug('Calculating Beta equilibrium for EOS')
                    be = BetaEquilibrium(self.ImportedEOS,
                                         np.linspace(0.7*kwargs['SkyrmeDensity']/0.16, 10, 100))
                else:
                    be = BetaEquilibrium(self.ImportedEOS,
                                         np.linspace(0.7*kwargs['SkyrmeDensity']/0.16, 1.5*kwargs['PolyTropeDensity']/0.16, 100))

                self.BENuclear = be[0]
                self.rho = be[1] 
                self.pfrac = be[2] 
                self.mufrac = be[3]   
                self.energy = be[4]
            else:
                self.rho = meta_data['rho']
                self.pfrac = meta_data['pfrac']
                self.mufrac = meta_data['mufrac']
                self.energy = meta_data['energy']
                self.BENuclear = sky.SplineEOS.Construct(self.rho, self.energy)
        else:
            self.BENuclear = self.ImportedEOS

        # Needs to fix maximum mass for the equation of state
        if EOSType == 'EOS' or EOSType == '3Poly' or EOSType == 'Rod' or EOSType == 'Power':
            if not 'PressureHigh' in kwargs:
                logger.debug('Looking for polytrope that forms designated neutron stars')
                kwargs['PressureHigh'] = 500
                try:
                    kwargs = self._FindMaxMassForEOS(**kwargs)
                except Exception as error:
                    logger.exception('Polytrope cannot help EOS attend the required maximum mass')
                    raise error

        return self.GetEOSType(**kwargs)

    def GetMetaData(self):
        return {'rho': self.rho, 'pfrac': self.pfrac, 'mufrac': self.mufrac, 'energy': self.energy}

    def GetEOSType(self, **kwargs):
        self.density_list = []
        self.eos_list = []
 
        EOSType = kwargs['EOSType']
        if EOSType == 'EOS' or EOSType == '3Poly' or EOSType == 'Rod' or EOSType == 'Power':
            self.GetEOS(**kwargs)
        elif EOSType == 'EOS2Poly' or EOSType == 'Meta2Poly': 
            self.GetEOS2Poly(**kwargs)
        elif EOSType == 'EOSNoPolyTrope' or EOSType == 'PowerNoPolyTrope' or EOSType == 'Meta':
            self.GetEOSNoPolyTrope(**kwargs)
        elif EOSType == 'EOSNoCrust':
            self.GetEOSNoCrust(**kwargs)
        self.density_list[-1] = (self.density_list[-1][0], 100)
        eos = sky.EOSConnect(self.density_list, self.eos_list)
        # also returns the arguments such that next time some calculation needs not be done
        return eos, [rho[0] for rho in self.density_list[1:][::-1]], kwargs

    def GetEOS(self, 
               CrustFileName, 
               CrustSmooth, 
               PressureHigh, 
               PolyTropeDensity, 
               TranDensity, 
               SkyrmeDensity, 
               **kwargs):
        self.InsertCrust(CrustFileName, TranDensity, CrustSmooth=CrustSmooth)
        self.InsertConnection(sky.PseudoEOS, SkyrmeDensity)
        self.InsertMain(self.BENuclear, PolyTropeDensity)
        self.InsertConnection(sky.PolyTrope, 7*self.ImportedEOS.rho0, final_pressure=PressureHigh)
        self.Finalize()

    def GetEOS2Poly(self, 
                    CrustFileName, 
                    CrustSmooth, 
                    PolyTropeDensity, 
                    TranDensity, 
                    SkyrmeDensity, 
                    **kwargs):
        self.InsertCrust(CrustFileName, TranDensity, CrustSmooth=CrustSmooth)
        self.InsertSmoothConnection(sky.PseudoEOS, SkyrmeDensity)
        self.InsertMain(self.BENuclear, PolyTropeDensity)
        self.InsertConnection(sky.ConstSpeed, 100)
        self.Finalize()

    def GetEOSNoPolyTrope(self, 
                          CrustFileName, 
                          CrustSmooth, 
                          TranDensity, 
                          SkyrmeDensity, 
                          **kwargs):
        self.InsertCrust(CrustFileName, TranDensity, CrustSmooth=CrustSmooth)
        self.InsertSmoothConnection(sky.PseudoEOS, SkyrmeDensity)
        self.InsertMain(self.BENuclear, 100)
        self.Finalize()


    def GetEOSNoCrust(self, PolyTropeDensity, **kwargs):
        self.InsertMain(self.BENuclear, 7*self.ImportedEOS.rho0)# PolyTropeDensity)
        #self.InsertConnection(sky.PolyTrope, 7*self.ImportedEOS.rho0, final_pressure=PressureHigh)
        self.Finalize()


    def _FindMaxMassForEOS(self, **kwargs):
        if not 'MaxMassRequested' in kwargs:
            kwargs['MaxMassRequested'] = 2.

        eos, _, _ = self.GetEOSType(**kwargs)
        def FixMaxMass(pressure):
            eos.eos_list[-1].ChangeFinalPressure(7*self.ImportedEOS.rho0, pressure)
            with wrapper.TidalLoveWrapper(eos) as tidal_love:
                result = tidal_love.FindMaxMass()
                pc = result['PCentral'] 
                max_m = result['mass']
                if np.isnan(max_m):
                    raise ValueError('Mass maximization failed')
            return max_m - kwargs['MaxMassRequested']

        kwargs['PressureHigh'] = opt.newton(FixMaxMass, x0=500, rtol=0.0001, tol=0.0001)
        return kwargs


    def InsertCrust(self, CrustFilename, final_density, **kwargs):
         crust = pd.read_csv(CrustFilename)
         self.crustEOS = sky.SplineEOS.Construct(crust['rho(fm-3)'].values,
                                                 energy_density=crust['E(MeV/fm3)'].values,
                                                 smooth=kwargs['CrustSmooth'],
                                                 pressure=crust['P(MeV/fm3)'].values)
         if len(self.density_list) != 0:
             raise RuntimeError('InsertCrust can only be used when it is the outermost layer. Are you sure that not EOS is inserted before Crust?')
         self.density_list.append((-1, final_density))
         self.eos_list.append(self.crustEOS)

    def InsertConnection(self, eos_type, final_density, **kwargs):
         if len(self.density_list) == 0:
             raise RuntimeError('Connection EOS cannot be placed at the start.')
         ini_density = self.density_list[-1][1]
         self.density_list.append((ini_density, final_density))
         self.eos_list.append((eos_type.MatchBothEnds, kwargs))

    def InsertSmoothConnection(self, eos_type, final_density, **kwargs):
         """
         Smooth connection is always spline EOS
         It takes any EOS type, but they will be all casted into Spline for smooth connection
         """
         if len(self.density_list) == 0:
             raise RuntimeError('Connection EOS cannot be placed at the start.')
         ini_density = self.density_list[-1][1]
         self.density_list.append((ini_density, final_density))
         kwargs['int_EOS_type'] = eos_type
         self.eos_list.append((sky.SplineEOS.SmoothConnection, kwargs))

    def InsertMain(self, eos, final_density):
         if len(self.density_list) > 0:
             ini_density = self.density_list[-1][1]
         else:
             ini_density = -1
         self.density_list.append((ini_density, final_density))
         self.eos_list.append(eos)

    def Finalize(self):
         for index, ((ini_density, final_density), prap_eos) in enumerate(zip(self.density_list, self.eos_list)):
             if isinstance(prap_eos, tuple):
                 constructor = prap_eos[0]
                 kwargs = prap_eos[1]
                 prev_eos = self.eos_list[index-1]
                 if index +1 >= len(self.eos_list):
                     next_eos = None
                 else:
                     next_eos = self.eos_list[index+1]
                 self.eos_list[index] = constructor(ini_rho=ini_density, 
                                                    ini_eos=prev_eos, 
                                                    final_rho=final_density, 
                                                    final_eos=next_eos, 
                                                    **kwargs)

            
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
    df = pd.read_csv('Results/OrigNew.csv', index_col=0)
    df.fillna(0, inplace=True)
 
    creator = EOSCreator(df.iloc[1])
    creator.PrepareEOS(**df.iloc[1])
    eos_connect, _ = creator.GetEOSType(**df.iloc[1])


    rho = np.linspace(0,10,500)

    """
    plt.plot(rho, eos_connect.GetEnergyDensity(rho*rho0, 0.), label='spline')
    plt.plot(rho, Nuclear.GetEnergy(rho*rho0, 0.), label='LNS')
    plt.ylabel(r'$Energy density (MeV/fm^{3})$')
    plt.xlabel(r'$\rho/\rho_{0}$')
    plt.legend()
    plt.show()
    """
    
    rho = np.concatenate([np.linspace(1e-12, 3.76e-4, 1000), np.linspace(3.77e-4, 2, 9000)])
    plt.plot(rho, eos_connect.GetPressure(rho, 0.), label='spline')
    plt.ylabel(r'$Pressure (MeV/fm^{3})$')
    plt.xlabel(r'$\rho/\rho_{0}$')
    plt.legend()
    plt.xlim([1e-7, 1])
    plt.ylim([1e-12, 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    plt.plot(eos_connect.GetEnergyDensity(rho, 0), eos_connect.GetPressure(rho, 0.), label='spline', color='b')
    plt.ylabel(r'$Pressure (MeV/fm^{3})$')
    plt.xlabel(r'$Energy density (MeV/fm^{3})$')
    plt.legend()
    plt.xlim([1e-8, 10])
    plt.ylim([1e-13, 100])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    #plt.plot(Nuclear.GetEnergyDensity(rho*rho0, 0), np.sqrt(Nuclear.GetSpeedOfSound(rho*rho0, 0)), 'ro', label='sound LNS', color='orange')
    plt.plot(eos_connect.GetEnergyDensity(rho, 0), np.sqrt(eos_connect.GetSpeedOfSound(rho, 0)), label='connected EOS real speed of sound', color='black')
    plt.xlabel(r'Energy Density (MeV/fm^{3})$')
    plt.ylabel(r'$Speed of sound$')
    plt.legend()
    plt.show()
    
