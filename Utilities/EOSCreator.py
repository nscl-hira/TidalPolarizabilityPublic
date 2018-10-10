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


    def __init__(self, row, **kwargs):#TranDensity=0.2355e-3, SkyrmeDensity=0.3*0.16, PolyTropeDensity=3*0.16, PressureHigh=1000, PRCTransDensity=None, CrustSmooth=0.0, **kwargs):
        if 'CrustFileName' in kwargs:
            crust_file_name = kwargs['CrustFileName']
        else:
            crust_file_name = 'Constraints/EOSCrustOutput.dat'
        crust = pd.read_csv(crust_file_name)#'Constraints/EOSCrustOutput.dat')
        self.crustEOS = sky.EOSSpline(crust['rho(fm-3)'], energy_density=crust['E(MeV/fm3)'], smooth=kwargs['CrustSmooth'], pressure=crust['P(MeV/fm3)'])
        self.kwargs = kwargs
        self.row = row
        if 'PolyHighP' in kwargs:
            self.PressureHigh = kwargs['PolyHighP']
        else:
            self.PressureHigh = None
        if 'SoundHighDensity' in kwargs:
            self.SoundHighDensity = kwargs['SoundHighDensity']
 
        # we assume that skyrme is included if a parameter named t0 is present
        if 't0' in row:
            Skyrme = sky.Skryme(row)
            self.Skyrme = Skyrme
            self.BENuclear = BetaEquilibrium(self.Skyrme)
        #self.BENuclear = Skyrme
        # if PRCTransDensity is larger than 0, we will use this value and overrwrite all custom TranDensity and SkyrmeDensity
        if kwargs['PRCTransDensity'] > 0: 
            self.kwargs['TranDensity'] = kwargs['PRCTransDensity']*FindCrustalTransDensity(Skyrme)
            self.kwargs['SkyrmeDensity'] = FindCrustalTransDensity(Skyrme)

    def GetCrust(self):
        return self.crustEOS, [self.kwargs['SkyrmeDensity'], self.kwargs['self.TranDensity']]

    def Get3Poly(self):
        TranDensity = self.kwargs['TranDensity']
        Pressure1 = self.row['Pressure1']
        rho1 = 0.67*0.16
        Pressure2 = self.row['Pressure2']
        rho2 = 3*0.16
        if self.PressureHigh is None:
            Pressure3 = self.row['Pressure3']
        else:
            Pressure3 = self.PressureHigh
        rho3 = 7*rho1

        poly1 = sky.PolyTrope(TranDensity,
                              self.crustEOS.GetEnergy(TranDensity, 0), 
                              self.crustEOS.GetAutoGradPressure(TranDensity, 0),
                              rho1, Pressure1)
        poly2 = sky.PolyTrope( rho1,
                               poly1.GetEnergy(rho1, 0),
                               poly1.GetAutoGradPressure(rho1, 0),
                               rho2, Pressure2)
        poly3 = sky.PolyTrope( rho2,
                               poly2.GetEnergy(rho2, 0),
                               poly2.GetAutoGradPressure(rho2, 0),
                               rho3, Pressure3)
        eos = sky.EOSConnect([(-1, TranDensity), 
                              (TranDensity, rho1), 
                              (rho1, rho2), 
                              (rho2, 100)], 
                             [self.crustEOS, poly1, poly2, poly3])

        return eos, [rho2, rho1, TranDensity]
       


    def GetEOS(self):
        # ONLY PSEUDOEOS TAKES ENERGY DENSITY! ALL OTHER EOS USES ENERGY
        TranDensity = self.kwargs['TranDensity']
        SkyrmeDensity = self.kwargs['SkyrmeDensity'] 
        PolyTropeDensity = self.kwargs['PolyTropeDensity']
        if self.PressureHigh is None:
            PressureHigh = self.kwargs['PressureHigh']
        else:
            PressureHigh = self.PressureHigh

        pseudo = sky.PseudoEOS(TranDensity, 
                               self.crustEOS.GetEnergyDensity(TranDensity, 0), 
                               self.crustEOS.GetAutoGradPressure(TranDensity, 0), 
                               SkyrmeDensity, 
                               self.BENuclear.GetEnergyDensity(SkyrmeDensity, 0), 
                               self.BENuclear.GetAutoGradPressure(SkyrmeDensity, 0))
        poly = sky.PolyTrope(PolyTropeDensity, 
                             self.BENuclear.GetEnergy(PolyTropeDensity, 0), 
                             self.BENuclear.GetAutoGradPressure(PolyTropeDensity, 0), 
                             7*self.Skyrme.rho0, PressureHigh)
        
        #return sky.EOSConnect([(0, 10)], [BENuclear])
        eos = sky.EOSConnect([(-1, TranDensity), 
                              (TranDensity, SkyrmeDensity), 
                              (SkyrmeDensity, PolyTropeDensity), 
                              (PolyTropeDensity, 100)], 
                             [self.crustEOS, pseudo, self.BENuclear, poly])
        return eos, [PolyTropeDensity, SkyrmeDensity, TranDensity]

    def GetEOSNoPolyTrope(self):
        # ONLY PSEUDOEOS TAKES ENERGY DENSITY! ALL OTHER EOS USES ENERGY
        TranDensity = self.kwargs['TranDensity']
        SkyrmeDensity = self.kwargs['SkyrmeDensity'] 


        pseudo = sky.PseudoEOS(TranDensity, 
                               self.crustEOS.GetEnergyDensity(TranDensity, 0), 
                               self.crustEOS.GetAutoGradPressure(TranDensity, 0), 
                               SkyrmeDensity, 
                               self.BENuclear.GetEnergyDensity(SkyrmeDensity, 0), 
                               self.BENuclear.GetAutoGradPressure(SkyrmeDensity, 0))
        eos = sky.EOSConnect([(-1, TranDensity), 
                              (TranDensity, SkyrmeDensity), 
                              (SkyrmeDensity, 100)], 
                             [self.crustEOS, pseudo, self.BENuclear])
        return eos, [SkyrmeDensity, TranDensity]

    def GetBESkyrme(self):
        TranDensity = self.kwargs['TranDensity']
        SkyrmeDensity = self.kwargs['SkyrmeDensity']
        return self.BENuclear, [SkyrmeDensity, TranDensity]

    def GetOnlySkyrme(self):
        TranDensity = self.kwargs['TranDensity']
        SkyrmeDensity = self.kwargs['SkyrmeDensity']
        return self.Skyrme, [SkyrmeDensity, TranDensity]

    def GetEOS2Poly(self):
        TranDensity = self.kwargs['TranDensity']
        SkyrmeDensity = self.kwargs['SkyrmeDensity'] 
        PolyTropeDensity = self.kwargs['PolyTropeDensity']

        poly1 = sky.PolyTrope(PolyTropeDensity, 
                              self.BENuclear.GetEnergy(PolyTropeDensity, 0), 
                              self.BENuclear.GetAutoGradPressure(PolyTropeDensity, 0), 
                              1, 1, gamma=14)
        poly2 = sky.ConstSpeed(self.SoundHighDensity,
                               poly1.GetEnergy(self.SoundHighDensity, 0),
                               poly1.GetAutoGradPressure(self.SoundHighDensity, 0),
                               speed_of_sound=0.8)

        # ONLY PSEUDOEOS TAKES ENERGY DENSITY! ALL OTHER EOS USES ENERGY
        pseudo = sky.PseudoEOS(TranDensity, 
                               self.crustEOS.GetEnergyDensity(TranDensity, 0), 
                               self.crustEOS.GetAutoGradPressure(TranDensity, 0), 
                               SkyrmeDensity, 
                               self.BENuclear.GetEnergyDensity(SkyrmeDensity, 0), 
                               self.BENuclear.GetAutoGradPressure(SkyrmeDensity, 0))
        
        eos = sky.EOSConnect([(-1, TranDensity), 
                              (TranDensity, SkyrmeDensity), 
                              (SkyrmeDensity, PolyTropeDensity), 
                              (PolyTropeDensity, self.SoundHighDensity), (self.SoundHighDensity, 100)],
                              [self.crustEOS, pseudo, self.BENuclear, poly1, poly2])
        return eos, [self.SoundHighDensity, PolyTropeDensity, SkyrmeDensity, TranDensity]

 
    """
    max_mass is there to fix the max mass of NS possible for Type=EOS
    It will be ignored for all other options
    I can do this with EOS because the upper limit of polytrope is a free parameter
    """
    def PrepareEOS(self, Type='EOS', max_mass=2):
        if Type == 'EOS':# or Type == '3Poly':
            pc_max = [0]
            def FixMaxMass(pressure):
                self.PressureHigh = pressure
                eos, _ = self.GetEOSType(Type)
                tidal_love = wrapper.TidalLoveWrapper(eos)

                calculated_max_mass, pc = tidal_love.FindMaxMass()
                if np.isnan(calculated_max_mass):
                    raise ValueError('Mass maximization failed')
                pc_max[0] = pc
                tidal_love.Close()
                return calculated_max_mass - max_mass
            pressure_high = opt.newton(FixMaxMass, x0=500)
            self.PressureHigh = pressure_high
            return {'PolyHighP': pressure_high}

        elif Type == 'EOS2Poly':
            TranDensity = self.kwargs['TranDensity']
            SkyrmeDensity = self.kwargs['SkyrmeDensity'] 
            PolyTropeDensity = self.kwargs['PolyTropeDensity']

            poly1 = sky.PolyTrope(PolyTropeDensity, 
                                  self.BENuclear.GetEnergy(PolyTropeDensity, 0), 
                                  self.BENuclear.GetAutoGradPressure(PolyTropeDensity, 0), 
                                  1, 1, gamma=14)
            # find where speed of sound = 95% c
            try:
                self.SoundHighDensity = opt.newton(lambda x: poly1.GetSpeedOfSound(x, 0) - 0.8*0.8, x0=PolyTropeDensity)   
                if self.SoundHighDensity < PolyTropeDensity:
                    self.SoundHighDensity = PolyTropeDensity
            except RuntimeError:
                raise ValueError('Cannot find density corresponds to 0.8c. This can be an indication that the starting energy/pressure is negative')
            return {'SoundHighDensity': self.SoundHighDensity}

        else:
            return {}
            

    def GetEOSType(self, Type='EOS'):
        """
        Avaliable choices are:
        EOS
        EOSNoPolyTrope
        BESkyrme
        OnlySkyrme
        """
        if Type == "EOS":
            return self.GetEOS()
        elif Type == "EOS2Poly":
            return self.GetEOS2Poly()
        elif Type == "EOSNoPolyTrope": 
            return self.GetEOSNoPolyTrope()
        elif Type == "BESkyrme":
            return self.GetBESkyrme()
        elif Type == '3Poly':
            return self.Get3Poly()
        else: 
            return self.GetOnlySkyrme()
            


if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_summary.csv', index_col=0)
    df.fillna(0, inplace=True)
 
    Nuclear = sky.Skryme(df.loc['LNS'])
    eos_connect = EOSCreator(Nuclear, SkyrmeDensity=0.3*rho0).GetEOS()
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
    plt.plot(rho, eos_connect.GetAutoGradPressure(rho, 0.), label='spline')
    plt.plot(rho, Nuclear.GetAutoGradPressure(rho, 0.), label='LNS')
    plt.ylabel(r'$Pressure (MeV/fm^{3})$')
    plt.xlabel(r'$\rho/\rho_{0}$')
    plt.legend()
    plt.xlim([1e-7, 1])
    plt.ylim([1e-12, 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    plt.plot(eos_connect.GetEnergyDensity(rho, 0), eos_connect.GetAutoGradPressure(rho, 0.), label='spline', color='b')
    plt.plot(Nuclear.GetEnergyDensity(rho, 0), Nuclear.GetAutoGradPressure(rho, 0.), label='LNS')
    plt.ylabel(r'$Pressure (MeV/fm^{3})$')
    plt.xlabel(r'$Energy density (MeV/fm^{3})$')
    plt.legend()
    plt.xlim([1e-8, 10])
    plt.ylim([1e-13, 100])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    spl = UnivariateSpline(eos_connect.GetEnergyDensity(rho, 0), eos_connect.GetAutoGradPressure(rho, 0.), s=0)
    connect_d = spl.derivative(1)

    spl = UnivariateSpline(Nuclear.GetEnergyDensity(rho, 0), Nuclear.GetAutoGradPressure(rho, 0.), s=0)
    nuclear_d = spl.derivative(1)

    plt.plot(eos_connect.GetEnergyDensity(rho, 0), np.sqrt(connect_d(eos_connect.GetEnergyDensity(rho, 0))), 'ro', label='connected EOS', color='b')
    plt.plot(Nuclear.GetEnergyDensity(rho, 0), np.sqrt(nuclear_d(Nuclear.GetEnergyDensity(rho, 0))), 'ro', label='Skyrme')
    #plt.plot(Nuclear.GetEnergyDensity(rho*rho0, 0), np.sqrt(Nuclear.GetSpeedOfSound(rho*rho0, 0)), 'ro', label='sound LNS', color='orange')
    plt.plot(eos_connect.GetEnergyDensity(rho, 0), np.sqrt(eos_connect.GetSpeedOfSound(rho, 0)), label='connected EOS real speed of sound', color='black')
    plt.xlabel(r'Energy Density (MeV/fm^{3})$')
    plt.ylabel(r'$Speed of sound$')
    plt.legend()
    plt.show()
    
