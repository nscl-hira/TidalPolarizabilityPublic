import tempfile
import TidalLove.TidalLove_individual as tidal
from decimal import Decimal
import autograd.numpy as np
import scipy.optimize as opt
import math
import logging
from multiprocessing_logging import install_mp_handler

logger = logging.getLogger(__name__)
install_mp_handler(logger)

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
            logger.debug('Write EOS into temp file')
            self.output = tempfile.NamedTemporaryFile('w')
        else:
            logger.debug('Write EOS into file %s' % name)
            self.output = open(name, 'w')
        eos.ToFileStream(self.output)
        self.max_energy, self.max_pressure = eos.GetMaxDef()
        logger.debug('EOS is valid up till energy = %f, pressure = %f' % (self.max_energy, self.max_pressure))
        # pressure needs to be expressed as pascal for pc
        self.max_pressure# /= 3.62704e-5
        self.ans = ()
        self.surface_pressure = 1e-8 # default pressure defined at surface
        self.checkpoint = [1e-3]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.Close()

    def Calculate(self, pc):
        # return order
        # m r lambda_ checkpt_m checkpt_r
        self.ans = tidal.tidallove_individual(self.output.name, 
                                              pc, self.max_energy, self.surface_pressure, np.array(self.checkpoint), )
        if(len(self.ans[4]) > 0):
            self.ans = (self.ans[0], self.ans[4][-1], self.ans[2], self.ans[3], self.ans[4])
        if self.ans[0] < 0:
            raise RuntimeError('Calculated mass smaller than zero. EOS exceed its valid range')
        return self.ans

    def FindMaxMass(self, central_pressure0=500, disp=False, *args):
        # checkpoint list must be in desending order
        self.checkpoint.sort(reverse=True)
         
        if central_pressure0 > self.max_pressure:
            logger.warning('Default pressure %g exceed max. valid pressure %.3f. Will ignore default pressure' % (central_pressure0, self.max_pressure))
            central_pressure0 = 0.7*self.max_pressure
        # try finding the maximum mass
        try:
            pc = opt.minimize(lambda x: -1e6*self.Calculate(float(x))[0], 
                              x0=np.array([central_pressure0]), 
                              bounds=((0, None),), 
                              options={'eps':0.1, 'ftol':1e-3})
        except Exception as error:
            logger.exception('Failed to find max mass')
            pc = {'x': [np.nan]}
        return (pc['x'][0],) +self.ans

    def FindMass(self, central_pressure0=60, mass=1.4, *args, **kwargs):
        # checkpoint list must be in desending order
        self.checkpoint.sort(reverse=True)

        if central_pressure0 > self.max_pressure:
            logger.warning('Default pressure %g exceed max. valid pressure %.3f. Will ignore default pressure' % (central_pressure0, self.max_pressure))
            central_pressure0 = 0.7*self.max_pressure
        try:
            pc = opt.newton(lambda x: self.Calculate(x)[0] - mass, 
                            x0=central_pressure0, *args, **kwargs)
        except Exception as error:
            logger.exception('Failed to find NS mass %g' % mass)
            pc = np.nan
            self.ans = tuple([np.nan for ans in self.ans])
        return (pc,) + self.ans 

    def Close(self):

        self.output.close()
    
