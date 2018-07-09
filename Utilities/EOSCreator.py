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
    
    return -3.75e-4*Skryme.GetL(rho0) + 0.0963 # fm-3


class EOSCreator:


    def __init__(self, Skyrme, TranDensity=0.2355e-3, SkyrmeDensity=0.3*rho0, PolyTropeDensity=3*rho0, PressureHigh=1000, PRCTransDensity=None, CrustSmooth=0.0, **kwargs):
        crust = pd.read_csv('Constraints/EOSCrustOutput.dat')
        self.crustEOS = sky.EOSSpline(crust['rho(fm-3)'], energy_density=crust['E(MeV/fm3)'], smooth=CrustSmooth, pressure=crust['P(MeV/fm3)'])
 
        self.Skyrme = Skyrme
        self.BENuclear = BetaEquilibrium(Skyrme)
        #self.BENuclear = Skyrme
        if PRCTransDensity:
            self.TranDensity = PRCTransDensity*FindCrustalTransDensity(Skyrme)
            self.SkyrmeDensity = FindCrustalTransDensity(Skyrme)
        else:
            self.TranDensity = TranDensity
            self.SkyrmeDensity = SkyrmeDensity
        self.PolyTropeDensity = PolyTropeDensity
        self.PressureHigh = PressureHigh

    def GetCrust(self):
        return self.crustEOS, [self.SkyrmeDensity, self.TranDensity]


    def GetEOS(self):
        # ONLY PSEUDOEOS TAKES ENERGY DENSITY! ALL OTHER EOS USES ENERGY
        pseudo = sky.PseudoEOS(self.TranDensity, 
                               self.crustEOS.GetEnergyDensity(self.TranDensity, 0), 
                               self.crustEOS.GetAutoGradPressure(self.TranDensity, 0), 
                               self.SkyrmeDensity, 
                               self.BENuclear.GetEnergyDensity(self.SkyrmeDensity, 0), 
                               self.BENuclear.GetAutoGradPressure(self.SkyrmeDensity, 0))
        poly = sky.PolyTrope(self.PolyTropeDensity, 
                             self.BENuclear.GetEnergy(self.PolyTropeDensity, 0), 
                             self.BENuclear.GetAutoGradPressure(self.PolyTropeDensity, 0), 
                             7*rho0, self.PressureHigh)
        
        #return sky.EOSConnect([(0, 10)], [BENuclear])
        eos = sky.EOSConnect([(-1, self.TranDensity), 
                              (self.TranDensity, self.SkyrmeDensity), 
                              (self.SkyrmeDensity, self.PolyTropeDensity), 
                              (self.PolyTropeDensity, 100)], 
                             [self.crustEOS, pseudo, self.BENuclear, poly])
        return eos, [self.PolyTropeDensity, self.SkyrmeDensity, self.TranDensity]

    def GetEOSNoPolyTrope(self):
        # ONLY PSEUDOEOS TAKES ENERGY DENSITY! ALL OTHER EOS USES ENERGY
        pseudo = sky.PseudoEOS(self.TranDensity, 
                               self.crustEOS.GetEnergyDensity(self.TranDensity, 0), 
                               self.crustEOS.GetAutoGradPressure(self.TranDensity, 0), 
                               self.SkyrmeDensity, 
                               self.BENuclear.GetEnergyDensity(self.SkyrmeDensity, 0), 
                               self.BENuclear.GetAutoGradPressure(self.SkyrmeDensity, 0))
        eos = sky.EOSConnect([(-1, self.TranDensity), 
                              (self.TranDensity, self.SkyrmeDensity), 
                              (self.SkyrmeDensity, 100)], 
                             [self.crustEOS, pseudo, self.BENuclear])
        return eos, [self.SkyrmeDensity, self.TranDensity]

    def GetBESkyrme(self):
        return self.BENuclear, [self.SkyrmeDensity, self.TranDensity]

    def GetOnlySkyrme(self):
        return self.Skyrme, [self.SkyrmeDensity, self.TranDensity]

    def GetEOS2Poly(self):
        poly1 = sky.PolyTrope(self.PolyTropeDensity, 
                              self.BENuclear.GetEnergy(self.PolyTropeDensity, 0), 
                              self.BENuclear.GetAutoGradPressure(self.PolyTropeDensity, 0), 
                              1, 1, gamma=14)
        # find where speed of sound = 95% c
        try:
            
            density = opt.newton(lambda x: poly1.GetSpeedOfSound(x, 0) - 0.95*0.95, x0=self.PolyTropeDensity)   
            if density < self.PolyTropeDensity:
                density = self.PolyTropeDensity
            poly2 = sky.ConstSpeed(density,
                                   poly1.GetEnergy(density, 0),
                                   poly1.GetAutoGradPressure(density, 0),
                                   speed_of_sound=0.95)
        except RuntimeError:
            raise ValueError('Cannot find density corresponds to 0.95c. This can be an indication that the starting energy/pressure is negative')
        

        # ONLY PSEUDOEOS TAKES ENERGY DENSITY! ALL OTHER EOS USES ENERGY
        pseudo = sky.PseudoEOS(self.TranDensity, 
                               self.crustEOS.GetEnergyDensity(self.TranDensity, 0), 
                               self.crustEOS.GetAutoGradPressure(self.TranDensity, 0), 
                               self.SkyrmeDensity, 
                               self.BENuclear.GetEnergyDensity(self.SkyrmeDensity, 0), 
                               self.BENuclear.GetAutoGradPressure(self.SkyrmeDensity, 0))
        
        eos = sky.EOSConnect([(-1, self.TranDensity), 
                              (self.TranDensity, self.SkyrmeDensity), 
                              (self.SkyrmeDensity, self.PolyTropeDensity), 
                              (self.PolyTropeDensity, density), (density, 100)],
                              [self.crustEOS, pseudo, self.BENuclear, poly1, poly2])
        return eos, [density, self.PolyTropeDensity, self.SkyrmeDensity, self.TranDensity]

 
    """
    max_mass is there to fix the max mass of NS possible for Type=EOS
    It will be ignored for all other options
    I can do this with EOS because the upper limit of polytrope is a free parameter
    """
    def PrepareEOS(self, Type='EOS', max_mass=2):
        if Type == 'EOS':
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
            return pressure_high, pc_max[0]
            

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
    
