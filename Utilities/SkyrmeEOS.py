import copy
from decimal import Decimal
import tempfile
import types
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
import math
import scipy.misc as misc
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import logging
from multiprocessing_logging import install_mp_handler, MultiProcessingHandler

#logger = logging.getLogger(__name__)
#install_mp_handler(logger)

from Utilities.Constants import *

# register interpolation methods into autograd
from autograd.extend import primitive, defvjp
@primitive
def interpolate(x, interpolator, deriv_order):
    return interpolator(x, nu=deriv_order)

def interpolate_vjp(ans, x, interpolator, deriv_order):
  return lambda g: g*interpolate(x, interpolator, deriv_order + 1)

defvjp(interpolate, interpolate_vjp)

class EOS:

    def __init__(self):
        self.rho0 = 0.16 # default value for satuation density is 0.16 fm-3

    """
    minimal EOS
    As long as Energy and Pressure vs rho is given, TOV equation will be able to use EOS
    Override GetEnergy for it to work
    If your pressure cannot be calculated by autograd, then override GetPressure as well
    If sound is not provided, it will be calculated with autograd, but if it cannot be calculated with autograd then you need to override it as well
    """
    def GetEnergy(self, rho, pfrac=0):
        pass

    def GetPressure(self, rho, pfrac=0):
        grad_edensity = egrad(self.GetEnergy, 0)(rho, pfrac)
        return rho*rho*grad_edensity

    def GetdPressure(self, rho, pfrac=0):
        return egrad(self.GetPressure, 0)(rho, pfrac)

    def GetEnergyDensity(self, rho, pfrac=0):
        return rho*self.GetEnergy(rho, pfrac)

    def GetSpeedOfSound(self, rho, pfrac=0):
        #return egrad(self.GetPressure, 0)(rho, pfrac)/(self.GetEnergy(rho, pfrac) + egrad(self.GetEnergy, 0)(rho, pfrac)*rho)
        return egrad(self.GetPressure, 0)(rho, pfrac)/egrad(self.GetEnergyDensity, 0)(rho, pfrac)

    def Get2EGrad(self, rho, pfrac=0):
        return egrad(egrad(self.GetEnergy, 0),0)(rho, pfrac)

    def ToFile(self, name):
        with open(name, 'w') as file_:
            self.ToFileStream(file_)

    def ToTempFile(self):
        with tempfile.NamedTemporaryFile('w') as file_:
            self.ToFileStream(file_)
            yield file_

    def ToFileStream(self, filestream):
        #print header
        filestream.write(" ========================================================\n")
        filestream.write("       E/V           P              n           eps      \n") 
        filestream.write("    (MeV/fm3)     (MeV/fm3)      (#/fm3)    (erg/cm^3/s) \n")
        filestream.write(" ========================================================\n")
        # the last 2 column (n and eps) is actually not used in the program
        # therefore eps column will always be zero
        n = np.concatenate([np.logspace(np.log(1e-10), np.log(3.76e-4), 2000, base=np.exp(1)), 
                            np.linspace(3.77e-4, 10*0.16, 18000)])
        energy = self.GetEnergyDensity(n, 0.)
        pressure = self.GetPressure(n, 0.)


        for density, e, p in zip(n, energy, pressure):
            if(not math.isnan(e) and not math.isnan(p)):
                filestream.write("   %.5e   %.5e   %.5e   0.0000e+0\n" % (Decimal(e), Decimal(p), Decimal(density)))
        filestream.flush()

    def GetMaxDef(self):
        # return the maximum energy at which the EOS is still monotonically increasing
        n = np.concatenate([np.logspace(np.log(1e-10), np.log(3.76e-4), 2000, base=np.exp(1)),
                            np.linspace(3.77e-4, 10*0.16, 18000)])
        energy = self.GetEnergyDensity(n, 0.)
        pressure = self.GetPressure(n, 0.)

        ediff = np.diff(energy) 
        pdiff = np.diff(pressure)

        if np.all(ediff > 0) and np.all(pdiff > 0):
            idx = -2
        else:
            idx = min(np.argmax(ediff < 0), np.argmax(pdiff < 0))
            #logger.warning('EOS stops increasing monotonically at index %d' % idx)
        emax = energy[idx]
        pmax = pressure[idx]
        return emax, pmax
            
        


    """
    Full EoS
    Where symmetric energy term is provided/ can be independently calculated
    Need to override GetAsymEnergy
    if autograd is not compatible with your GetAsymEnergy
    you need to override all other listed functions as well
    """
    

    def GetAsymEnergy(self, rho, *args):
        return 1./8.*egrad(egrad(self.GetEnergy, 1), 1)(rho, np.full(np.array(rho).shape, 0.5))


    def GetK(self, rho, pfrac=0):
        sec_grad_density = egrad(egrad(self.GetEnergy, 0), 0)(rho, pfrac)
        return 9*rho*rho*sec_grad_density
    
    def GetQ(self, rho, pfrac=0):
        """
        For some weird reasons, Q is also called -K' which we will use to compare
        """
        third_grad_density = egrad(egrad(egrad(self.GetEnergy, 0), 0), 0)(rho, pfrac)
        return 27*rho*rho*rho*third_grad_density

    def GetZ(self, rho, pfrac=0):
        forth_grad_density = egrad(egrad(egrad(egrad(self.GetEnergy, 0), 0), 0), 0)(rho, pfrac)
        return 81*rho*rho*rho*rho*forth_grad_density

    """
    Usually these functions are defined only at rho0
    When calling them, try to set rho to rho0
    """
    def GetL(self, rho):
        grad_S = egrad(self.GetAsymEnergy, 0)(rho)
        return 3*rho*grad_S
    
    def GetKsym(self, rho):
        second_grad_S = egrad(egrad(self.GetAsymEnergy, 0), 0)(rho)
        return 9*rho*rho*second_grad_S
    
    def GetQsym(self, rho):
        third_grad_S = egrad(egrad(egrad(self.GetAsymEnergy, 0), 0), 0)(rho)
        return 27*rho*rho*rho*third_grad_S

    def GetZsym(self, rho):
        forth_grad_density = egrad(egrad(egrad(egrad(self.GetAsymEnergy, 0), 0), 0), 0)(rho)
        return 81*rho*rho*rho*rho*forth_grad_density



class EOSConnect(EOS):
    """
    Connect EOS from different intervals to make a single EOS
    """

    def __init__(self, intervals, eos_list):
        EOS.__init__(self)
        self.eos_list = eos_list
        self.intervals = intervals

    def GetEnergy(self, rho, pfrac=0):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetEnergy(rho, pfrac))(eos) for eos in self.eos_list])

    def GetEnergyDensity(self, rho, pfrac=0):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetEnergyDensity(rho, pfrac))(eos) for eos in self.eos_list])

    def GetPressure(self, rho, pfrac=0):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetPressure(rho, pfrac))(eos) for eos in self.eos_list])

    def GetdPressure(self, rho, pfrac=0):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetdPressure(rho, pfrac))(eos) for eos in self.eos_list])


    def Get2EGrad(self, rho, pfrac=0):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.Get2EGrad(rho, pfrac))(eos) for eos in self.eos_list])

    def GetSpeedOfSound(self, rho, pfrac=0):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetSpeedOfSound(rho, pfrac))(eos) for eos in self.eos_list])

    def _Interval(self, rho):
        return [(rho > interval[0]) & (rho <= interval[1]) for interval in self.intervals]

"""
Connection EOS
Used only to connect 2 EOS from below and above
must provide classmethod MatchBothEnds
such that can be smooth out by Spline
"""

class PseudoEOS(EOS):

    """
    This EOS would not obey P = -rho^2de/drho
    The only reason it exists it to bridge different EOS together
    Since TOV equation only cares about energy as a function of pressure, 
    Density is irrelavent here
    Therefore we are ignoring the relation of variables with rho
    rho is only used as a parameter such that E vs P graph is contineous
    Speed of sound is now discontineous
    """

    def __init__(self, ini_rho, ini_energy_density, ini_pressure, final_rho, final_energy_density, final_pressure):
        EOS.__init__(self)       
        """
        EnergyDensity = A*rho + B
        A, B is chosen such that the energy at corresponding rho agree
        """
        self.A = (final_energy_density - ini_energy_density)/(final_rho - ini_rho)
        self.B = ini_energy_density - self.A*ini_rho

        """
        Assume ultra relativistic fermi gas
        P = C + K*energy_density^4/3
        """
        self.K = (final_pressure - ini_pressure)/(np.power(final_energy_density, 4./3.) - np.power(ini_energy_density, 4./3.))
        self.C = ini_pressure - self.K*np.power(ini_energy_density, 4./3.)

    def GetPressure(self, rho, pfrac=0):
        return self.C + self.K*np.power(self.GetEnergyDensity(rho, pfrac), 4./3.)

    def GetEnergy(self, rho, pfrac=0):
        return self.GetEnergyDensity(rho, pfrac)/rho

    def GetEnergyDensity(self, rho, pfrac=0):
        return self.A*rho + self.B

    def GetSpeedOfSound(self, rho, pfrac=0):
        return 4./3.*self.K*np.power(self.GetEnergyDensity(rho, 0), 1./3.)

    def Get2EGrad(self, rho, pfrac=0):
        pass

    @classmethod
    def MatchBothEnds(cls, ini_rho, ini_eos, final_rho, final_eos, **kwargs):
        ini_energy_density = ini_eos.GetEnergyDensity(ini_rho, 0)
        ini_pressure = ini_eos.GetPressure(ini_rho, 0)
        final_energy_density = final_eos.GetEnergyDensity(final_rho, 0)
        final_pressure = final_eos.GetPressure(final_rho, 0)

        return cls(ini_rho, ini_energy_density, ini_pressure, final_rho, final_energy_density, final_pressure)


class CubicEOS(EOS):

    """
    Similar to Pseudo EOS. Does not correspond to any physical process
    it exist such that 2 EOSs can be connected with smooth first derivative
    """

    def __init__(self, ini_rho, ini_eos, final_rho, final_eos):
        EOS.__init__(self)       
        """
        E = A + B*rho + C*rho^2 + D*rho^3 + E*rho^4 + F*rho^5
        """

        mat = np.array([[1, ini_rho, ini_rho**2, ini_rho**3, ini_rho**4, ini_rho**5],
                        [1, final_rho, final_rho**2, final_rho**3, final_rho**4, final_rho**5],
                        [0, 1, 2*ini_rho, 3*ini_rho**2, 4*ini_rho**3, 5*ini_rho**4],
                        [0, 1, 2*final_rho, 3*final_rho**2, 4*final_rho**3, 5*final_rho**4],
                        [0, 0, 2, 6*ini_rho, 12*ini_rho**2, 20*ini_rho**3],
                        [0, 0, 2, 6*final_rho, 12*final_rho**2, 20*final_rho**3]])
        pfrac = 0
        b = np.array([ini_eos.GetEnergy(ini_rho, pfrac), 
                      final_eos.GetEnergy(final_rho, pfrac), 
                      ini_eos.GetPressure(ini_rho, pfrac)/(ini_rho*ini_rho),
                      final_eos.GetPressure(final_rho, pfrac)/(final_rho*final_rho),
                      ini_eos.Get2EGrad(ini_rho, pfrac),
                      final_eos.Get2EGrad(final_rho, pfrac)])

        self.coeff = np.linalg.solve(mat, b)


    def GetEnergy(self, rho, pfrac=0):
        return self.coeff[0] + self.coeff[1]*rho + self.coeff[2]*np.power(rho, 2) + self.coeff[3]*np.power(rho, 3) + self.coeff[4]*np.power(rho, 4) + self.coeff[5]*np.power(rho, 5)

    @classmethod
    def MatchBothEnds(cls, ini_rho, ini_eos, final_rho, final_eos, **kwargs):
        return cls(ini_rho, ini_eos, final_rho, final_eos)

class PolyTrope(EOS):


    def __init__(self, init_density, init_energy, init_pressure, final_density, final_pressure, gamma=None):
        EOS.__init__(self)
        self.init_pressure = init_pressure
        self.init_energy = init_energy
        self.init_density = init_density
        if gamma is None:
            self.gamma = np.log(final_pressure/init_pressure)/np.log(final_density/init_density)
        else:
            self.gamma = gamma
        self.K = init_pressure/np.power(init_density, self.gamma)

    def GetEnergy(self, rho, pfrac=0):
        return self.init_energy + self.K*(np.power(rho, self.gamma-1) - np.power(self.init_density, self.gamma-1))/(self.gamma - 1)

    def ChangeFinalPressure(self, final_density, final_pressure):
        self.gamma = np.log(final_pressure/self.init_pressure)/np.log(final_density/self.init_density)
        self.K = self.init_pressure/np.power(self.init_density, self.gamma)

    @classmethod
    def MatchBothEnds(cls, ini_rho, ini_eos, final_rho, final_eos, **kwargs):
        ini_energy = ini_eos.GetEnergy(ini_rho)
        ini_pressure = ini_eos.GetPressure(ini_rho)
        return cls(ini_rho, ini_energy, ini_pressure, final_rho, **kwargs)
    

class ConstSpeed(EOS):
    
    def __init__(self, init_density, init_energy, init_pressure, speed_of_sound=0.95):
        EOS.__init__(self)
        self.init_pressure = init_pressure
        self.init_energy = init_energy
        self.init_density = init_density
        self.speed_of_sound = speed_of_sound
        self.C1 = (init_pressure + init_energy*init_density)/((speed_of_sound + 1)*np.power(init_density, speed_of_sound + 1))
        self.C2 = init_pressure - speed_of_sound*init_energy*init_density

    def GetEnergy(self, rho, pfrac=0):
        return self.C1*np.power(rho, self.speed_of_sound)  - self.C2/((self.speed_of_sound + 1)*rho)

    @classmethod
    def MatchBothEnds(cls, ini_rho, ini_eos, final_rho, final_eos, **kwargs):
        ini_energy = ini_eos.GetEnergy(ini_rho, 0)
        ini_pressure = ini_eos.GetPressure(ini_rho, 0)
        return cls(ini_rho, ini_energy, ini_pressure, **kwargs)
 

"""
EOS with real physical meaning
e.g. Skyrmes, Polytrope, Spline with other functionals...
"""

class SplineEOS(EOS):

    """
    Barebone EOSSpline where the use supply only energy as a function of rho0
    This class will calculate everything by itself
    """
    def __init__(self, rho, energy, smooth=0):
        EOS.__init__(self)
        self.spl = UnivariateSpline(rho, energy, s=smooth)
        #self.SplPressure = None
        #self.SymSpl = UnivariateSpline(rho, np.full(rho.shape, 0))
        #self.SplPressure = None

    def GetEnergy(self, rho, pfrac=0):
        return interpolate(rho, self.spl, 0)# + (2*pfrac - 1)**2*interpolate(rho, self.SymSpl, 0)

    #def GetPressure(self, rho, pfrac=0):
    #    if self.SplPressure is None:
    #        grad_edensity = egrad(self.GetEnergy, 0)(rho, pfrac)
    #        return rho*rho*grad_edensity
    #    else:
    #        return interpolate(rho, self.SplPressure, 0)

    def _SetPressure(self, rho, pressure, smooth):
        self.SplPressure = UnivariateSpline(rho, pressure, s=smooth)
        def CustomGetPressure(self, rho, pfrac=0):
            return interpolate(rho, self.SplPressure, 0)
        self.GetPressure = types.MethodType(CustomGetPressure, self)

    def _SetAsym(self, rho, Sym, smooth):
        self.SymSpl = UnivariateSpline(rho, Sym, s=smooth)
        def CustomGetEnergy(self, rho, pfrac=0):
            return interpolate(rho, self.spl, 0) + (2*pfrac - 1)**2*interpolate(rho, self.SymSpl, 0)
        self.GetEnergy = types.MethodType(CustomGetEnergy, self)

    @classmethod
    def Construct(cls, rho, energy=None, pressure=None, energy_density=None, rho_Sym=None, Sym=None, smooth=0):
        """
        Create a spline according to what you've provided. Bare minimum: energy as a function of rho
        """
        if energy is None:
            energy = energy_density/rho

        eos = cls(rho, energy, smooth)
        if pressure is not None:
            eos._SetPressure(rho, pressure, smooth)

        if Sym is not None:
            eos._SetAsym(rho_Sym, Sym, smooth)
        return eos

    @classmethod
    def SmoothConnection(cls, ini_rho, ini_eos, int_EOS_type, final_rho, final_eos, **kwargs):
        """
        Create a smooth transition between crustal EOS and Skyrme
        works by leaving gaps between the crustal EOS, intermediate pseudo EOS and Skyrme EOS
        and let Spline EOS interpolate between them
        Spline is 3rd degree, so the connection should be natually smooth
        """
        peos = int_EOS_type.MatchBothEnds(ini_rho, ini_eos, final_rho, final_eos, **kwargs)
    
        rho_range = final_rho - ini_rho
        start_rho = np.linspace(ini_rho, ini_rho + 0.1*rho_range, 10, endpoint=True)
        mid_rho = np.linspace(ini_rho + 0.3*rho_range, ini_rho + 0.4*rho_range, 0, endpoint=True)
        end_rho = np.linspace(final_rho, final_rho + 0.1*rho_range, 10, endpoint=True)
    
        energy = peos.GetEnergy(mid_rho, 0)
        pressure = peos.GetPressure(mid_rho, 0)
    
        energy = np.concatenate([ini_eos.GetEnergy(start_rho, 0), energy, final_eos.GetEnergy(end_rho, 0)])
        pressure = np.concatenate([ini_eos.GetPressure(start_rho,  0), pressure, final_eos.GetPressure(end_rho, 0)])
    
        eos = cls.Construct(np.concatenate([start_rho, mid_rho, end_rho]), energy=energy, pressure=pressure)
    
        def CustomGetSpeedOfSound(self, rho, pfrac=0, final_eos=final_eos):
            return final_eos.GetSpeedOfSound(rho, pfrac)
        eos.GetSpeedOfSound = types.MethodType(CustomGetSpeedOfSound, eos)
    
        return eos


   


class FermiGas(EOS):

  
    def __init__(self, mass):
        EOS.__init__(self)
        self.mass = mass

    def GetEnergyDensity(self, rho, pfrac=0):
        a = self.mass*self.mass
        b = hbar*hbar

        """
        reduce to a form of integrate x^2sqrt(a + b*x^2) dx from 0 to 3.094rho
        """
        def anti_deriv(x):
            #return (np.sqrt(b*(a+b*x*x))*x*(a+2*b*x*x)-a*a*np.log(np.sqrt(b*(a+b*x*x))+b*x))/(8*np.power(b, 1.5)*pi2)
            return (np.sqrt(b*(a+b*x*x))*x*(a+2*b*x*x)-a*a*np.log((np.sqrt(a+b*x*x)+hbar*x)/self.mass))/(8*np.power(b, 1.5)*pi2)

        return (anti_deriv(3.094*np.power(rho, 0.33333333333333)) - anti_deriv(0))

    def GetEnergy(self, rho, pfrac=0):
        return self.GetEnergyDensity(rho, pfrac)/rho


class PowerLawEOS(EOS):

    def __init__(self, para):
        super().__init__()
        self.Esat = para['Esat']
        self.Ksat = para['Ksat']
        self.Qsat = para['Qsat']
        self.Zsat = para['Zsat']
        
        self.Esym = para['Esym']
        self.Lsym = para['Lsym']
        self.Ksym = para['Ksym']
        self.Qsym = para['Qsym']
        self.Zsym = para['Zsym']
        self.rho0 = 0.16
        
    def GetEnergy(self, rho, pfrac=0):
        delta = 1 - 2*pfrac
        x = (rho - self.rho0)/(3*self.rho0)
        E_iso_sca = self.Esat + 1./2.*self.Ksat*x*x + 1./6.*self.Qsat*x*x*x + 1./24.*self.Zsat*x*x*x*x
        E_iso_vec = self.Esym + self.Lsym*x + 1./2.*self.Ksym*x*x + 1./6.*self.Qsym*x*x*x + 1./24.*self.Zsym*x*x*x*x
        return E_iso_sca + delta*delta*E_iso_vec + 938.27

class MetaModeling(EOS):
    def __init__(self, para):
        super().__init__()
        Esat = para['Esat']
        Ksat = para['Ksat']
        Qsat = para['Qsat']
        if 'Zsat' in para:
            Zsat = para['Zsat']
        else:
            Psym = para['Psym']
            if 'PsymDens' in para:
                PsymDens = para['PsymDens']
            else:
                PsymDens = 4
            Zsat = 0
            

        Esym = para['Esym']
        Lsym = para['Lsym']
        Ksym = para['Ksym']
        Qsym = para['Qsym']
        Zsym = para['Zsym']

        msat = para['msat']
        self.rho0 = 0.16

        self.tFG_sat = 22.1 # MeV
        self.ksat = 1/msat - 1

        if 'mn' in para and 'mp' in para:
            mn = para['mn']
            mp = para['mp']
            self.ksym = 0.5*(1/mn - 1/mp)
        else:
            self.ksym = self.ksat - para['kv']

        self.vis = np.zeros(5)
        self.vis[0] = Esat - self.tFG_sat*(1+self.ksat)
        self.vis[1] = -self.tFG_sat*(2+5*self.ksat)
        self.vis[2] = Ksat - 2*self.tFG_sat*(-1+5*self.ksat)
        self.vis[3] = Qsat - 2*self.tFG_sat*(4-5*self.ksat)
        self.vis[4] = Zsat - 8*self.tFG_sat*(-7+5*self.ksat)

        self.viv = np.zeros(5)
        self.viv[0] = Esym - 5./9.*self.tFG_sat*(1 + (self.ksat + 3*self.ksym))
        self.viv[1] = Lsym - 5./9.*self.tFG_sat*(2 + 5*(self.ksat + 3*self.ksym))
        self.viv[2] = Ksym - 10./9.*self.tFG_sat*(-1 + 5*(self.ksat + 3*self.ksym))
        self.viv[3] = Qsym - 10./9.*self.tFG_sat*(4 - 5*(self.ksat+3*self.ksym))
        self.viv[4] = Zsym - 40./9.*self.tFG_sat*(-7 + 5*(self.ksat + 3*self.ksym))

        if 'Zsat' not in para:
            from scipy.optimize import fsolve
            def func(Zsat):
                self.vis[4] = Zsat - 8*self.tFG_sat*(-7+5*self.ksat)
                return self.GetPressure(PsymDens*self.rho0, 0.5) - Psym
            root = fsolve(func, 0)
            Zsat = root[0]
            self.vis[4] = Zsat - 8*self.tFG_sat*(-7+5*self.ksat)

    def _f1(self, delta):
        return np.power(1+delta, 5./3.) + np.power(1-delta, 5./3.)

    def _f2(self, delta):
        return delta*(np.power(1+delta, 5./3.) - np.power(1-delta, 5./3.))

    def _Kinetic(self, rho, delta):
        return self.tFG_sat/2*np.power(rho/self.rho0, 2./3.)*((1+self.ksat*rho/self.rho0)*self._f1(delta) + self.ksym*rho/self.rho0*self._f2(delta))

    def _u_N_alpha(self, N, alpha, x, rho):
        b = 10*np.log(2)
        return 1 - np.power(-3*x, N+1-alpha)*np.exp(-b*rho/self.rho0)

    def GetEnergy(self, rho, pfrac):
        delta = 1 - 2*pfrac
        x = (rho - self.rho0)/(3*self.rho0)
        energy = self._Kinetic(rho, delta)
        for alpha in range(5):
            energy += (self.vis[alpha] + self.viv[alpha]*delta*delta)*np.power(x, alpha)*self._u_N_alpha(4, alpha, x, rho)/math.factorial(alpha)
        return energy + 938.27

class BillEOS(MetaModeling):
    def __init__(self, para):
        para = copy.deepcopy(para)
        self.rho01 = 0.1
        self.C = 12.5

        self.SINT0 = 0
        self.SINT1 = 0
        self.SINT2 = 0
        self.SINT3 = 0

        if 'SINT0' in para and 'SINT1' in para:
            self.SINT0 = para['SINT0']
            self.SINT1 = para['SINT1']
            if 'SINT2' in para:
                self.SINT2 = para['SINT2']
            else:
                self.SINT2 = 0 # will be fixed by condition S(0) = 0 if SINT2 is not provided
            self.SINT3 = 0 # will be fixed by condition S(0) = 0

            para['Esym'] = 0
            para['Lsym'] = 0
            para['Ksym'] = 0
            para['Qsym'] = 0
            para['Zsym'] = 0
            super().__init__(para)
            Sym0 = self.GetAsymEnergy(0)
            if 'SINT2' in para:
                self.SINT3 = -Sym0*6/np.power(-self.rho01, 3)
            else:
                self.SINT2 = -Sym0*2/np.power(-self.rho01, 2)
        elif 'Esym' in para and 'Lsym' in para and 'Ksym' in para:
            Esym = para['Esym']
            Lsym = para['Lsym']
            Ksym = para['Ksym']
            para['Esym'] = 0
            para['Lsym'] = 0
            para['Ksym'] = 0
            para['Qsym'] = 0
            para['Zsym'] = 0
            super().__init__(para)
            # convert rho0 parameters to rho01 parameters
            # 0                            = S01 + S'(-rho01)      + 1/2S''(rho01)^2       - 1/6S'''(rho01)^3 # S(0) = 0
            # S - A(rho/rho0)^2/3          = S01 + S'(rho - rho01) + 1/2S''(rho - rho01)^2 + 1/6S'''(rho - rho01)^3
            # dS - 2/3A/rho0^2/3rho^-1/3   =       S'              + S''(rho - rho01)      + 1/2S'''(rho - rho01)^2
            # d^2S + 2/9A/rho0^2/3rho^-4/3 =                         S''                   + S'''(rho - rho01)  
            a = np.array([[1, -self.rho01           , 0.5*self.rho01*self.rho01              , -1/6.*self.rho01*self.rho01*self.rho01],
                          [1, self.rho0 - self.rho01, 0.5*np.power(self.rho0 - self.rho01, 2), 1/6.*np.power(self.rho0 - self.rho01, 3)],
                          [0, 1                     , self.rho0 - self.rho01                 , 1/2.*np.power(self.rho0 - self.rho01, 2)],
                          [0, 0                     , 1                                      , self.rho0 - self.rho01]])
            b = np.array([0, 
                          Esym - self.C,
                          Lsym/(3*self.rho0) - 2/3.*self.C/self.rho0, 
                          Ksym/np.power(3*self.rho0, 2) + 2/9*self.C/self.rho0/self.rho0])
            x = np.linalg.solve(a, b)
            self.SINT0 = x[0]
            self.SINT1 = x[1]
            self.SINT2 = x[2]
            self.SINT3 = x[3]
        elif 'E01' in para and 'L01' in para and 'K01' in para:
            para['Esym'] = 0
            para['Lsym'] = 0
            para['Ksym'] = 0
            para['Qsym'] = 0
            para['Zsym'] = 0

            super().__init__(para)
            self.SINT0 = para['E01'] - self.C*np.power(self.rho01/self.rho0, 2/3.)
            self.SINT1 = para['L01']/(3*self.rho01) - 2/3.*self.C/np.power(self.rho0, 2/3.)/np.power(self.rho01, 1/3.)
            self.SINT2 = para['K01']/np.power(3*self.rho01, 2) + 2/9.*self.C/np.power(self.rho0, 2/3.)/np.power(self.rho01, 4/3.)
            self.SINT3 = 0
            Sym0 = self.GetAsymEnergy(0)
            self.SINT3 = -Sym0*6/np.power(-self.rho01, 3)
        else:
            raise Exception('Wrong input parameters')
                        


    def GetEnergy(self, rho, pfrac):
        delta = 1 - 2*pfrac
        symEnergy = super().GetEnergy(rho, pfrac=0.5)
        Skin = self.C*np.power(rho/self.rho0, 2/3.)
        Sint = self.SINT0 + self.SINT1*(rho - self.rho01) + 1/2*self.SINT2*np.power(rho - self.rho01, 2) + 1/6*self.SINT3*np.power(rho - self.rho01, 3)
        return symEnergy + delta*delta*(Skin + Sint)

 
class Skryme(EOS):



    def __init__(self, para):
        super().__init__()
        self.para = para
        self.a = para['t1']*(para['x1']+2) + para['t2']*(para['x2']+2)
        self.b = 0.5*(para['t2']*(2*para['x2']+1)-para['t1']*(2*para['x1']+1))
        if 'rho0' in para:
            self.rho0 = para['rho0']
        else:
            # default value is 0.16
            self.rho0 = 0.16

    def __GetH(self, n, pfrac):
        return (2**(n-1))*(pfrac**n + (1-pfrac)**n)
    
    """
    Return m* = M*/M instead of M*
    """
    def GetEffectiveMass(self, rho, pfrac=0):
        result = self.a*self.__GetH(5./3., pfrac) + self.b*self.__GetH(8./3., pfrac)
        result *= (mn*rho/(4*(hbar**2)))
        result += self.__GetH(5./3., pfrac)
        return 1./result

    def GetMs(self, rho):
        return 1./(1. + mn*rho*(self.a+self.b)/(4*(hbar**2)))

    def GetMv(self, rho):
        return 1./(1. + mn*rho*self.a/(4*(hbar**2)))

    def GetFI(self, rho):
        return 1./self.GetMs(rho) - 1./self.GetMv(rho)
    
    def GetEnergy(self, rho, pfrac=0):
        result = 3.*(hbar**2.)/(10.*mn)*((3.*pi2/2.)**0.666667)*np.power(rho, 0.6667)*self.__GetH(5./3., pfrac)
        result += self.para['t0']/8.*rho*(2.*(self.para['x0']+2.)-(2.*self.para['x0']+1)*self.__GetH(2., pfrac))
        for i in range(1, 4):
            result += 1./48.*self.para['t3%d'%i]*(rho**(self.para['sigma%d'%i]+1.))*(2.*(self.para['x3%d'%i]+2.)-(2.*self.para['x3%d'%i]+1.)*self.__GetH(2., pfrac))
        result += 3./40.*((3.*pi2/2.)**0.666667)*np.power(rho, 5./3.)*(self.a*self.__GetH(5./3., pfrac)+self.b*self.__GetH(8./3., pfrac))
        return result + mn
    
    def ToCSV(self, filename, rho, pfrac):

        sym_energy = self.GetAsymEnergy(rho)
        energy = self.GetEnergy(rho, pfrac)
        energy_density = self.GetEnergyDensity(rho, pfrac)
        pressure = self.GetPressure(rho, pfrac)
        data = np.vstack((sym_energy, energy, energy_density, pressure, rho, rho/self.rho0)).T
       
        with open(filename, 'w') as file_:
            file_.write('''({dashes}
{S:^12},{EV:^12},{EN:^12},{P:^12},{rho:^12},{rho_rho0}:^12
{dashes})\n'''.format(dashes='-'*106, S='S(rho) (MeV)', EV='E/V (MeV/fm3)', EN='E/N MeV', P='P (MeV/fm3)', rho='rho (fm-3)', rho_rho0='rho/rho0'))
            np.savetxt(file_, data, fmt='%12.3f', delimiter=',')


    


