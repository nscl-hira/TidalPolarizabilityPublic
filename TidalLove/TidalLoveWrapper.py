import tempfile
import TidalLove.TidalLove_individual as tidal
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
            self.output = tempfile.NamedTemporaryFile('w')
        else:
            self.output = open(name, 'w')
        eos.ToFileStream(self.output)
        self.ans = ()
        self.checkpoint = [0.1]

    def __enter__(self):
        return self

    def __exit__(self, type, value, trackback):
        self.Close()

    def Calculate(self, pc):
        # return order
        # m r lambda_ checkpt_m checkpt_r
        self.ans = tidal.tidallove_individual(self.output.name, 
                                              pc, np.array(self.checkpoint))
        if(len(self.ans[4]) > 0):
            self.ans = (self.ans[0], self.ans[4][-1], self.ans[2], self.ans[3], self.ans[4])
        return self.ans

    def FindMaxMass(self, central_pressure0=500, disp=False, *args):
        # checkpoint list must be in desending order
        self.checkpoint.sort(reverse=True)
        # try finding the maximum mass
        try:
            pc = opt.minimize(lambda x: -1e6*self.Calculate(float(x))[0], 
                              x0=np.array([central_pressure0]), 
                              bounds=((0, None),), 
                              options={'eps':0.1, 'ftol':1e-3})
        except RuntimeError as error:
            pc = {'x': np.nan}
        return (pc['x'][0],) +self.ans

    def FindMass(self, central_pressure0=60, mass=1.4, *args):
        # checkpoint list must be in desending order
        self.checkpoint.sort(reverse=True)
        try:
            pc = opt.newton(lambda x: self.Calculate(x)[0] - mass, 
                            x0=central_pressure0, *args)
        except RuntimeError as error:
            pc = np.nan
            self.ans = tuple([np.nan for ans in self.ans])
        return (pc,) + self.ans 

    def Close(self):
        self.output.close()
    
