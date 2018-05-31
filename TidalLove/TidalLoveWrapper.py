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
        self.tempfile = tempfile.NamedTemporaryFile()
        self.output = tempfile.NamedTemporaryFile()
        #print header
        self.output.write(" ========================================================\n")
        self.output.write("       E/V           P              n           eps      \n") 
        self.output.write("    (MeV/fm3)     (MeV/fm3)      (#/fm3)    (erg/cm^3/s) \n")
        self.output.write(" ========================================================\n")
        # the last 2 column (n and eps) is actually not used in the program
        # therefore eps column will always be zero
        n = np.linspace(1e-12, 2, 10000)
        energy = (self.eos.GetEnergyDensity(n, 0.))
        pressure = self.eos.GetAutoGradPressure(n, 0.) 
        for density, e, p in zip(n, energy, pressure):
            self.output.write("   %.5e   %.5e   %.5e   0.0000e+0\n" % (Decimal(e), Decimal(p), Decimal(density)))

    def Calculate(self, pc):
        mass, radius, lambda_ = tidal.tidallove_individual(self.output.name, pc)
        return mass, radius, lambda_

    

