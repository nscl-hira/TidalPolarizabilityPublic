from pebble import ProcessPool
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

from TidalLove import TidalLoveWrapper as wrapper
import Utilities as utl
import SkyrmeEOS as sky 
from Constants import *
from BetaEquilibrium import BetaEquilibrium


def FindCrustalTransDensity(Skryme):
    """
    This function uses result from PhysRevC.83.048510
    which contains a formula calculated empirically
    where density transition occurs
    Please refer to equation 17 - 24 in the paper for details
    """
    
    return -3.75e-4*Skryme.GetL(Skryme.rho0) + 0.0963 # fm-3


class EOSCreator:


    def __init__(self, row):
        self.EQType = None # type of EOS for which equilibrium is calculated
        self.row = row
        self.BENuclear = None 
        self.rho = None 
        self.pfrac = None 
        self.mufrac = None


    def PrepareEOS(self, **kwargs):
        EOSType = kwargs['EOSType']

        if EOSType == 'Rod':
            if self.EQType != 'Rod':
                # load Rodrigo EFT functional
                df_E = pd.read_csv('SkyrmeParameters/Rodrigo.csv')
                df_Sym = pd.read_csv('SkyrmeParameters/Rodrigo_sym.csv')
                self.Skyrme = sky.EOSSpline(df_E['rho(fm-3)'], energy=df_E['E(MeV/fm3)'] + 931.8, rho_Sym=df_Sym['rho(fm-3)'], Sym=df_Sym['Sym(MeV/fm3)'])
                self.EQType = 'Rod'
                self.BENuclear, self.rho, self.pfrac, self.mufrac = BetaEquilibrium(self.Skyrme)
        elif self.EQType != 'Skyrme':
            self.Skyrme = sky.Skryme(self.row)
            self.EQType = 'Skyrme'
            self.BENuclear, self.rho, self.pfrac, self.mufrac = BetaEquilibrium(self.Skyrme)

        if EOSType == "EOS" or EOSType == "EOS2Poly" or EOSType == "EOSNoCrust":
            if 'PolyTropeDensity' not in kwargs:
                kwargs['PolyTropeDensity'] = 3*0.16
        if EOSType == "Rod":
            if 'PolyTropeDensity' not in kwargs:
                kwargs['PolyTropeDensity'] = 1.5*0.16
        # List for creating crustal EoS
        if EOSType == "EOS" or EOSType == "EOS2Poly" or EOSType == "EOSNoPolyTrope" or EOSType == "Rod":
            if 'CrustSmooth' not in kwargs:
                kwargs['CrustSmooth'] = 0.
            if 'CrustFileName' not in kwargs:
                kwargs['CrustFileName'] = 'Constraints/EOSCrustOutput.dat' 
            crust = pd.read_csv(kwargs['CrustFileName'])
            self.crustEOS = sky.EOSSpline(crust['rho(fm-3)'], 
                                          energy_density=crust['E(MeV/fm3)'], 
                                          smooth=kwargs['CrustSmooth'], 
                                          pressure=crust['P(MeV/fm3)'])
            if kwargs['PRCTransDensity'] > 0:
                kwargs['TranDensity'] = kwargs['PRCTransDensity']*FindCrustalTransDensity(self.Skyrme)
                kwargs['SkyrmeDensity'] = FindCrustalTransDensity(self.Skyrme)
            elif 'TranDensity' not in kwargs or 'SkyrmeDensity' not in kwargs:
                print('Cannot proceed without transition density information for crustal EoS')
            
        if EOSType == '3Poly':
            if 'Pressure1' not in kwargs:
                kwargs['Pressure1'] = 10
            if 'Pressure2' not in kwargs:
                kwargs['Pressure2'] = 50

        # Needs to fix maximum mass for the equation of state
        if EOSType == 'EOS' or EOSType == '3Poly' or EOSType == 'EOSNoCrust' or EOSType == 'Rod':
            if not 'PressureHigh' in kwargs:
                if not 'MaxMassRequested' in kwargs:
                    kwargs['MaxMassRequested'] = 2.

                def FixMaxMass(pressure):
                    eos, _ = self.GetEOSType(PressureHigh=pressure, **kwargs)
                    with wrapper.TidalLoveWrapper(eos) as tidal_love:
                        pc, max_m, _, _, _, _ = tidal_love.FindMaxMass()
                        if np.isnan(max_m):
                            raise ValueError('Mass maximization failed')
                    return max_m - kwargs['MaxMassRequested']

                kwargs['PressureHigh'] = opt.newton(FixMaxMass, x0=500)

        # Set speed of sound 
        if EOSType == 'EOS2Poly':
            if 'SoundSpeed' not in kwargs:
                kwargs['SoundSpeed'] = 0.8
            if 'SoundHighDensity' not in kwargs:
                PolyTropeDensity = kwargs['PolyTropeDensity']
                poly1 = sky.PolyTrope(PolyTropeDensity, 
                                      self.BENuclear.GetEnergy(kwargs['PolyTropeDensity'], 0), 
                                      self.BENuclear.GetPressure(kwargs['PolyTropeDensity'], 0), 
                                      1, 1, gamma=14)
                # find where speed of sound = 95% c
                try:
                    SoundHighDensity = opt.newton(lambda x: poly1.GetSpeedOfSound(x, 0) 
                                                  - kwargs['SoundSpeed']*kwargs['SoundSpeed'], 
                                                       x0=PolyTropeDensity)   
                    if SoundHighDensity < PolyTropeDensity:
                        SoundHighDensity = PolyTropeDensity
                except RuntimeError:
                    raise ValueError('Cannot find density corresponds to 0.8c. This can be an indication that the starting energy/pressure is negative')
                kwargs['SoundHighDensity'] = SoundHighDensity
        return kwargs


    def GetCrust(self, **kwargs):
        return self.crustEOS, []

    def Get3Poly(self, **kwargs):
        TranDensity = kwargs['TranDensity']
        Pressure1 = kwargs['Pressure1']
        Pressure2 = kwargs['Pressure2']
        Pressure3 = kwargs['PressureHigh']

        rho1 = 0.67*0.16
        rho2 = 3*0.16
        rho3 = 7*rho1

        poly1 = sky.PolyTrope(TranDensity,
                              self.crustEOS.GetEnergy(TranDensity, 0), 
                              self.crustEOS.GetPressure(TranDensity, 0),
                              rho1, Pressure1)
        poly2 = sky.PolyTrope( rho1,
                               poly1.GetEnergy(rho1, 0),
                               poly1.GetPressure(rho1, 0),
                               rho2, Pressure2)
        poly3 = sky.PolyTrope( rho2,
                               poly2.GetEnergy(rho2, 0),
                               poly2.GetPressure(rho2, 0),
                               rho3, Pressure3)
        eos = sky.EOSConnect([(-1, TranDensity), 
                              (TranDensity, rho1), 
                              (rho1, rho2), 
                              (rho2, 100)], 
                             [self.crustEOS, poly1, poly2, poly3])

        return eos, [rho2, rho1, TranDensity]
       
    def GetEOSNoCrust(self, **kwargs):
        PolyTropeDensity = kwargs['PolyTropeDensity']
        PressureHigh = kwargs['PressureHigh']

        poly = sky.PolyTrope(PolyTropeDensity, 
                             self.BENuclear.GetEnergy(PolyTropeDensity, 0), 
                             self.BENuclear.GetPressure(PolyTropeDensity, 0), 
                             7*self.Skyrme.rho0, PressureHigh)
        
        #return sky.EOSConnect([(0, 10)], [BENuclear])
        eos = sky.EOSConnect([(-1, PolyTropeDensity), 
                              (PolyTropeDensity, 100)], 
                             [self.Skyrme, poly])
        return eos, [PolyTropeDensity]


    def GetEOS(self, **kwargs):
        # ONLY PSEUDOEOS TAKES ENERGY DENSITY! ALL OTHER EOS USES ENERGY
        TranDensity = kwargs['TranDensity']
        SkyrmeDensity = kwargs['SkyrmeDensity'] 
        PolyTropeDensity = kwargs['PolyTropeDensity']
        PressureHigh = kwargs['PressureHigh']

        """
        pseudo = sky.PseudoEOS(TranDensity, 
                               self.crustEOS.GetEnergyDensity(TranDensity, 0), 
                               self.crustEOS.GetPressure(TranDensity, 0), 
                               SkyrmeDensity, 
                               self.BENuclear.GetEnergyDensity(SkyrmeDensity, 0), 
                               self.BENuclear.GetPressure(SkyrmeDensity, 0))
        """
        pseudo = sky.SmoothPseudo(TranDensity,
                                  self.crustEOS,
                                  SkyrmeDensity,
                                  self.BENuclear)
        poly = sky.PolyTrope(PolyTropeDensity, 
                             self.BENuclear.GetEnergy(PolyTropeDensity, 0), 
                             self.BENuclear.GetPressure(PolyTropeDensity, 0), 
                             7*self.Skyrme.rho0, PressureHigh)
        
        #return sky.EOSConnect([(0, 10)], [BENuclear])
        eos = sky.EOSConnect([(-1, TranDensity), 
                              (TranDensity, SkyrmeDensity), 
                              (SkyrmeDensity, PolyTropeDensity), 
                              (PolyTropeDensity, 100)], 
                             [self.crustEOS, pseudo, self.BENuclear, poly])
        return eos, [PolyTropeDensity, SkyrmeDensity, TranDensity]

    def GetEOSNoPolyTrope(self, **kwargs):
        # ONLY PSEUDOEOS TAKES ENERGY DENSITY! ALL OTHER EOS USES ENERGY
        TranDensity = kwargs['TranDensity']
        SkyrmeDensity = kwargs['SkyrmeDensity'] 

        """
        pseudo = sky.PseudoEOS(TranDensity, 
                               self.crustEOS.GetEnergyDensity(TranDensity, 0), 
                               self.crustEOS.GetPressure(TranDensity, 0), 
                               SkyrmeDensity, 
                               self.BENuclear.GetEnergyDensity(SkyrmeDensity, 0), 
                               self.BENuclear.GetPressure(SkyrmeDensity, 0))
        """
        pseudo = sky.SmoothPseudo(TranDensity,
                                  self.crustEOS,
                                  SkyrmeDensity,
                                  self.BENuclear)

        eos = sky.EOSConnect([(-1, TranDensity), 
                              (TranDensity, SkyrmeDensity), 
                              (SkyrmeDensity, 100)], 
                             [self.crustEOS, pseudo, self.BENuclear])
        return eos, [SkyrmeDensity, TranDensity]

    def GetBESkyrme(self, **kwargs):
        TranDensity = kwargs['TranDensity']
        SkyrmeDensity = kwargs['SkyrmeDensity']
        return self.BENuclear, [SkyrmeDensity, TranDensity]

    def GetOnlySkyrme(self, **kwargs):
        TranDensity = kwargs['TranDensity']
        SkyrmeDensity = kwargs['SkyrmeDensity']
        return self.Skyrme, []

    def GetEOS2Poly(self, **kwargs):
        TranDensity = kwargs['TranDensity']
        SkyrmeDensity = kwargs['SkyrmeDensity'] 
        PolyTropeDensity = kwargs['PolyTropeDensity']
        SoundHighDensity = kwargs['SoundHighDensity']

        poly1 = sky.PolyTrope(PolyTropeDensity, 
                              self.BENuclear.GetEnergy(PolyTropeDensity, 0), 
                              self.BENuclear.GetPressure(PolyTropeDensity, 0), 
                              1, 1, gamma=14)
        poly2 = sky.ConstSpeed(SoundHighDensity,
                               poly1.GetEnergy(SoundHighDensity, 0),
                               poly1.GetPressure(SoundHighDensity, 0),
                               speed_of_sound=0.8)

        # ONLY PSEUDOEOS TAKES ENERGY DENSITY! ALL OTHER EOS USES ENERGY
        pseudo = sky.PseudoEOS(TranDensity, 
                               self.crustEOS.GetEnergyDensity(TranDensity, 0), 
                               self.crustEOS.GetPressure(TranDensity, 0), 
                               SkyrmeDensity, 
                               self.BENuclear.GetEnergyDensity(SkyrmeDensity, 0), 
                               self.BENuclear.GetPressure(SkyrmeDensity, 0))
        
        eos = sky.EOSConnect([(-1, TranDensity), 
                              (TranDensity, SkyrmeDensity), 
                              (SkyrmeDensity, PolyTropeDensity), 
                              (PolyTropeDensity, SoundHighDensity), (SoundHighDensity, 100)],
                              [self.crustEOS, pseudo, self.BENuclear, poly1, poly2])
        #print([TranDensity, SkyrmeDensity, PolyTropeDensity])
        #print([self.crustEOS.GetEnergy(TranDensity, 0), self.BENuclear.GetEnergy(SkyrmeDensity, 0)])
        """
        rho = np.linspace(1e-4, 1, 1000)
        pressure = eos.GetPressure(rho, 0)
        energy = eos.GetEnergyDensity(rho, 0)
        plt.plot(energy, pressure)
        plt.show()
        plt.plot(rho, energy)
        plt.show()
        plt.plot(rho, pressure)
        plt.show()
        """
        return eos, [SoundHighDensity, PolyTropeDensity, SkyrmeDensity, TranDensity]

 
    def GetEOSType(self, EOSType='EOS', **kwargs):
        """
        Avaliable choices are:
        EOS
        EOSNoPolyTrope
        BESkyrme
        OnlySkyrme
        """
        if EOSType == "EOS":
            return self.GetEOS(**kwargs)
        elif EOSType == "EOS2Poly":
            return self.GetEOS2Poly(**kwargs)
        elif EOSType == "EOSNoPolyTrope": 
            return self.GetEOSNoPolyTrope(**kwargs)
        elif EOSType == "BESkyrme":
            return self.GetBESkyrme(**kwargs)
        elif EOSType == '3Poly':
            return self.Get3Poly(**kwargs)
        elif EOSType == 'EOSNoCrust':
            return self.GetEOSNoCrust(**kwargs)
        elif EOSType == 'Rod':
            return self.GetEOS(**kwargs) # you should have supplied Rodrigo EOS
        else: 
            return self.GetOnlySkyrme(**kwargs)
            


if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_summary.csv', index_col=0)
    df.fillna(0, inplace=True)
 
    Nuclear = sky.Skryme(df.loc['BSK10'])
    eos_connect = EOSCreator(Nuclear).GetEOSType('EOSNoCrust')
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
    plt.plot(rho, Nuclear.GetPressure(rho, 0.), label='LNS')
    plt.ylabel(r'$Pressure (MeV/fm^{3})$')
    plt.xlabel(r'$\rho/\rho_{0}$')
    plt.legend()
    plt.xlim([1e-7, 1])
    plt.ylim([1e-12, 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    plt.plot(eos_connect.GetEnergyDensity(rho, 0), eos_connect.GetPressure(rho, 0.), label='spline', color='b')
    plt.plot(Nuclear.GetEnergyDensity(rho, 0), Nuclear.GetPressure(rho, 0.), label='LNS')
    plt.ylabel(r'$Pressure (MeV/fm^{3})$')
    plt.xlabel(r'$Energy density (MeV/fm^{3})$')
    plt.legend()
    plt.xlim([1e-8, 10])
    plt.ylim([1e-13, 100])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    spl = UnivariateSpline(eos_connect.GetEnergyDensity(rho, 0), eos_connect.GetPressure(rho, 0.), s=0)
    connect_d = spl.derivative(1)

    spl = UnivariateSpline(Nuclear.GetEnergyDensity(rho, 0), Nuclear.GetPressure(rho, 0.), s=0)
    nuclear_d = spl.derivative(1)

    plt.plot(eos_connect.GetEnergyDensity(rho, 0), np.sqrt(connect_d(eos_connect.GetEnergyDensity(rho, 0))), 'ro', label='connected EOS', color='b')
    plt.plot(Nuclear.GetEnergyDensity(rho, 0), np.sqrt(nuclear_d(Nuclear.GetEnergyDensity(rho, 0))), 'ro', label='Skyrme')
    #plt.plot(Nuclear.GetEnergyDensity(rho*rho0, 0), np.sqrt(Nuclear.GetSpeedOfSound(rho*rho0, 0)), 'ro', label='sound LNS', color='orange')
    plt.plot(eos_connect.GetEnergyDensity(rho, 0), np.sqrt(eos_connect.GetSpeedOfSound(rho, 0)), label='connected EOS real speed of sound', color='black')
    plt.xlabel(r'Energy Density (MeV/fm^{3})$')
    plt.ylabel(r'$Speed of sound$')
    plt.legend()
    plt.show()
    
