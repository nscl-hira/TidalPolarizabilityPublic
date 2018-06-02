import tempfile
import TidalLove_individual as tidal
from decimal import Decimal
import autograd.numpy as np

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
        # When density is 0, everything = 0
        # unfortunately this simple point cannot be handled by regular eos calculation
        # So we are doing this manually
        # self.output.write("1e-15 1e-15 1e-15 0\n")
        # the last 2 column (n and eps) is actually not used in the program
        # therefore eps column will always be zero
        n = np.concatenate([np.linspace(1e-12, 3.76e-4, 1000), np.linspace(3.77e-4, 2, 9000)])#np.linspace(1e-12, 2, 10000) 
        energy = (self.eos.GetEnergyDensity(n, 0.))
        pressure = self.eos.GetAutoGradPressure(n, 0.) 
        for density, e, p in zip(n, energy, pressure):
            self.output.write("   %.5e   %.5e   %.5e   0.0000e+0\n" % (Decimal(e), Decimal(p), Decimal(density)))


    def Calculate(self, pc):
        mass, radius, lambda_ = tidal.tidallove_individual(self.output.name, pc)
        return mass, radius, lambda_


    def Close(self):
        self.output.close()
    

