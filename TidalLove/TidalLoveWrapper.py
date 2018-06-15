import tempfile
import TidalLove_individual as tidal
from decimal import Decimal
import autograd.numpy as np
import scipy.optimize as opt

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *

   
class TidalLoveWrapper:


    def __init__(self, eos):
        """
        Print the selected EOS into a file for the tidallove script to run
        """
        self.eos = eos
        self.output = tempfile.NamedTemporaryFile()
        #print header
        self.output.write(" ========================================================\n")
        self.output.write("       E/V           P              n           eps      \n") 
        self.output.write("    (MeV/fm3)     (MeV/fm3)      (#/fm3)    (erg/cm^3/s) \n")
        self.output.write(" ========================================================\n")
        # the last 2 column (n and eps) is actually not used in the program
        # therefore eps column will always be zero
        # n = np.concatenate([np.linspace(3e-7, 3.76e-4, 1000), np.linspace(3.77e-4, 5, 9000)])#np.linspace(1e-12, 2, 10000) 
        n = np.concatenate([np.logspace(np.log(1e-10), np.log(3.76e-4), 1000, base=np.exp(1)), np.linspace(3.77e-4, 5, 9000)])#np.linspace(1e-12, 2, 10000) 
        energy = (self.eos.GetEnergyDensity(n, 0.))
        pressure = self.eos.GetAutoGradPressure(n, 0.) 
        for density, e, p in zip(n, energy, pressure):
            self.output.write("   %.5e   %.5e   %.5e   0.0000e+0\n" % (Decimal(e), Decimal(p), Decimal(density)))
       
        self.mass = 0

    def Calculate(self, pc):
        mass, radius, lambda_ = tidal.tidallove_individual(self.output.name, pc)
        return mass, radius, lambda_

    def FindMaxMass(self, central_pressure0=30, disp=False, *args):
        # try finding the maximum mass
        pc = np.nan
        try:
            pc = opt.fmin(self._GetMass14, x0=central_pressure0, disp=disp, *args)
            max_mass = self.mass#, _, _ = self.Calculate(pc[0])
        except RuntimeError as error:
            max_mass = np.nan
        return max_mass, pc

    def FindMass14(self, central_pressure0=30, *args):
        try:
            pc = opt.newton(self._GetMass14, x0=central_pressure0, *args)
            mass, radius, lambda_ = self.Calculate(pc)
        except RuntimeError as error:
            mass, radius, lambda_, pc = np.nan, np.nan, np.nan, np.nan
        return mass, radius, lambda_, pc

    def Close(self):
        self.output.close()
    
    def _GetMass14(self, pc):
        mass, _, _ = self.Calculate(pc)
        self.mass = mass
        return -mass + 1.4
