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
        #self.max_pressure# /= 3.62704e-5
        self.ans = ()
        self.surface_pressure = 1e-8 # default pressure defined at surface
        self._checkpoint = [self.surface_pressure]
        self._named_density_checkpoint = []
        self._density_checkpoint = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.Close()

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, value):
        # checkpoint list must be in desending order
        value.sort(reverse=True)
        self._density_checkpoint = []
        for pre in value:
            try:
                self._density_checkpoint.append(opt.newton(lambda rho: self.eos.GetPressure(rho, 0) - pre, x0=self.eos.rho0, 
                                                           fprime=lambda x: self.eos.GetdPressure(x, 0)))
            except Exception:
                self._density_checkpoint.append(0)
        self._checkpoint = value

    @property
    def density_checkpoint(self):
        return self._density_checkpoint

    @density_checkpoint.setter
    def density_checkpoint(self, value):
        value.sort(reverse=True)
        self._checkpoint = self.eos.GetPressure(np.array(value), 0).tolist()
        self._density_checkpoint = value

    @property
    def named_density_checkpoint(self):
        return self._named_density_checkpoint
 
    @named_density_checkpoint.setter
    def named_density_checkpoint(self, value):
        # named checkpoints must be list of tuple
        value.sort(reverse=True, key=lambda tup: tup[1])
        self._named_density_checkpoint = value
        self.density_checkpoint = [val[1] for val in value]

    def Calculate(self, pc):
        # return order
        # m r lambda_ checkpt_m checkpt_r
        ans = tidal.tidallove_individual(self.output.name, 
                                         pc, self.max_energy, self.surface_pressure, np.array(self.checkpoint), )
        #if(len(ans[4]) > 0):
        self.ans = {'mass': ans[0], 
                    'Radius': ans[1], 
                    'Lambda': ans[2], 
                    'Checkpoint_mass': ans[3], 
                    'Checkpoint_radius': ans[4]}
        if self.ans['mass'] < 0:
            nan_arr = np.empty(ans[3].shape)
            nan_arr.fill(np.nan)
            self.ans = {'mass': np.nan, 
                        'Radius': np.nan, 
                        'Lambda': np.nan, 
                        'Checkpoint_mass': nan_arr, 
                        'Checkpoint_radius': nan_arr}
            raise RuntimeError('Calculated mass smaller than zero. EOS exceed its valid range')

        return self.ans

    def FindMaxMass(self, central_pressure0=500, disp=False, *args):
        if central_pressure0 > self.max_pressure:
            logger.warning('Default pressure %g exceed max. valid pressure %.3f. Will ignore default pressure' % (central_pressure0, self.max_pressure))
            central_pressure0 = 0.7*self.max_pressure
        # try finding the maximum mass
        try:
            pc = opt.minimize(lambda x: -1e6*self.Calculate(float(x))['mass'], 
                              x0=np.array([central_pressure0]), 
                              bounds=((0, None),), 
                              options={'eps':0.1, 'ftol':1e-3})
            pc = pc['x'][0]
        except Exception as error:
            logger.exception('Failed to find max mass')
            pc = np.nan
        # infer central density from central pressure
        try:
            DensCentralMax = opt.newton(lambda x: self.eos.GetPressure(x, 0) - pc, x0=5*0.16,
                                        fprime=lambda x: self.eos.GetdPressure(x, 0)) 
        except Exception as error:
            logger.exception('Cannot find central density for mass %g' % self.ans['mass'])
            DensCentralMax = np.nan
        return {'PCentral': pc, 'DensCentral':DensCentralMax, **self.ans}

    def FindMass(self, central_pressure0=60, mass=1.4, *args, **kwargs):
        if central_pressure0 > self.max_pressure:
            logger.warning('Default pressure %g exceed max. valid pressure %.3f. Will ignore default pressure' % (central_pressure0, self.max_pressure))
            central_pressure0 = 0.7*self.max_pressure
        try:
            pc = opt.newton(lambda x: self.Calculate(x)['mass'] - mass, 
                            x0=central_pressure0, *args, **kwargs)
        except Exception as error:
            logger.exception('Failed to find NS mass %g' % mass)
            pc = np.nan

        try:
            DensCentral = opt.newton(lambda x: self.eos.GetPressure(x, 0) - pc, x0=1.5*0.16,
                                     fprime=lambda x: self.eos.GetdPressure(x, 0)) 
        except Exception as error:
            logger.exception('Cannot find central density for mass %g' % mass)
            DensCentral = np.nan

        return {'PCentral': pc, 'DensCentral':DensCentral, **self.ans}

    def Close(self):
        self.output.close()
    
