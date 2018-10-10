import tempfile
import TidalLove_individual as tidal
from decimal import Decimal
import autograd.numpy as np
import scipy.optimize as opt
import math

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *

   
class TidalLoveWrapper:


    def __init__(self, eos, name=None):
        """
        Print the selected EOS into a file for the tidallove script to run
        """
        self.eos = eos
        if name is None:
            self.output = tempfile.NamedTemporaryFile()
        else:
            self.output = open('AllSkyrmes/%s.csv' % name, 'w') 
        #print header
        self.output.write(" ========================================================\n")
        self.output.write("       E/V           P              n           eps      \n") 
        self.output.write("    (MeV/fm3)     (MeV/fm3)      (#/fm3)    (erg/cm^3/s) \n")
        self.output.write(" ========================================================\n")
        # the last 2 column (n and eps) is actually not used in the program
        # therefore eps column will always be zero
        # n = np.concatenate([np.linspace(3e-7, 3.76e-4, 1000), np.linspace(3.77e-4, 5, 9000)])#np.linspace(1e-12, 2, 10000) 
        n = np.concatenate([np.logspace(np.log(1e-10), np.log(3.76e-4), 2000, base=np.exp(1)), np.linspace(3.77e-4, 10, 18000)])
        #n = np.linspace(1e-12, 10, 20000) 
        energy = (self.eos.GetEnergyDensity(n, 0.))
        pressure = self.eos.GetAutoGradPressure(n, 0.) 
        for density, e, p in zip(n, energy, pressure):
            if(not math.isnan(e) and not math.isnan(p)):
                self.output.write("   %.5e   %.5e   %.5e   0.0000e+0\n" % (Decimal(e), Decimal(p), Decimal(density)))
       
        self.output.flush()
        self.mass = 0
        self.target_mass = 1.4
        self.checkpoint = []

    def Calculate(self, pc):
        mass, radius, lambda_, checkpoint_mass, checkpoint_radius = tidal.tidallove_individual(self.output.name, pc, np.array(self.checkpoint))
        if(len(checkpoint_mass) > 0):
            radius = checkpoint_radius[-1]
        return mass, radius, lambda_, checkpoint_mass, checkpoint_radius

    def FindMaxMass(self, central_pressure0=500, disp=False, *args):
        # try finding the maximum mass
        pc = {'x': np.nan}
        try:
            # constraint on causality
            """
            def MaxSpeedOfSound(pressure):
                #rho = np.linspace(1e-3, 5, 300)
                #pressure_list = self.eos.GetAutoGradPressure(rho, 0)
                # find the cross over point
                def GetDensityFromPressure(rho):
                    return self.eos.GetAutoGradPressure(rho, 0) - pressure
                density = opt.newton(GetDensityFromPressure, x0=7*0.16)
                
                if np.isnan(density):
                    raise RuntimeError('Causality not satisfied')  
                return - self.eos.GetSpeedOfSound(density, 0) + 1

            # optimize
            con = {'type': 'ineq', 'fun': MaxSpeedOfSound}
            pc = opt.minimize(lambda x: 1e6*self._GetMass14(x), x0=np.array([central_pressure0]), method='SLSQP', constraints=con, options={'eps':0.1, 'ftol':1e-8})
            """
            pc = opt.minimize(lambda x: 1e6*self._GetMass14(x), x0=np.array([central_pressure0]), bounds=((0, None),), options={'eps':0.1, 'ftol':1e-8})
            max_mass = self.mass#, _, _ = self.Calculate(pc[0])
        except RuntimeError as error:
            max_mass = np.nan
        return max_mass, pc['x'][0]

    def FindMass(self, central_pressure0=60, mass=1.4, *args):
        try:
            self.target_mass = mass
            pc = opt.newton(self._GetMass14, x0=central_pressure0, *args)
            mass, radius, lambda_, checkpoint_mass, checkpoint_radius = self.Calculate(pc)
        except RuntimeError as error:
            mass, radius, lambda_, pc, checkpoint_mass, checkpoint_radius = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return mass, radius, lambda_, pc, checkpoint_mass, checkpoint_radius

    def Close(self):
        self.output.close()
    
    def _GetMass14(self, pc):
        pc = float(pc)
        mass, _, _, _, _ = self.Calculate(pc)
        self.mass = mass
        return -mass + self.target_mass
