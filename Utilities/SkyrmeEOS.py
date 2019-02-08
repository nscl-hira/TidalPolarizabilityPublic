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

from Constants import *

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
    def GetEnergy(self, rho, pfrac):
        pass

    def GetPressure(self, rho, pfrac):
        grad_edensity = egrad(self.GetEnergy, 0)(rho, pfrac)
        return rho*rho*grad_edensity

    def GetEnergyDensity(self, rho, pfrac):
        return rho*self.GetEnergy(rho, pfrac)

    def GetSpeedOfSound(self, rho, pfrac):
        return egrad(self.GetPressure, 0)(rho, pfrac)/(self.GetEnergy(rho, pfrac) + egrad(self.GetEnergy, 0)(rho, pfrac)*rho)

    def Get2EGrad(self, rho, pfrac):
        return egrad(egrad(self.GetEnergy, 0),0)(rho, pfrac)

    def ToFile(self, name):
        with open(name, 'w') as file_:
            self.ToFileStream(file_)

    def ToTempFile(self):
        with tempfile.NamedTemporaryFile() as file_:
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
        energy = (self.GetEnergyDensity(n, 0.))
        pressure = self.GetPressure(n, 0.) 
        for density, e, p in zip(n, energy, pressure):
            if(not math.isnan(e) and not math.isnan(p)):
                filestream.write("   %.5e   %.5e   %.5e   0.0000e+0\n" % (Decimal(e), Decimal(p), Decimal(density)))
        filestream.flush()


class SplineEOS(EOS):

    """
    Barebone EOSSpline where the use supply only energy as a function of rho0
    This class will calculate everything by itself
    """
    def __init__(self, rho, energy, smooth=0):
        EOS.__init__(self)
        self.spl = UnivariateSpline(rho, energy, s=smooth)
        self.dspl = self.spl.derivative(1)
        self.ddspl = self.spl.derivative(2)

    def GetEnergy(self, rho, pfrac):
        return self.spl(rho)

    def GetEnergyDensity(self, rho, pfrac):
        return rho*self.GetEnergy(rho, pfrac)

    def GetPressure(self, rho, pfrac):
        grad_edensity = self.dspl(rho)
        pressure = rho*rho*grad_edensity
        return pressure

    def Get2EGrad(self, rho, pfrac):
        return self.ddspl(rho)

    def GetSpeedOfSound(self, rho, pfrac):
        return (2*rho*(self.dspl(rho)) + rho*rho*(self.ddspl(rho))) \
               /(self.GetEnergy(rho, pfrac) + self.dspl(rho)*rho)

class SplineEOSFull(SplineEOS):

    def __init__(self, rho, energy, rho_Sym, Sym, smooth=0):
        SplineEOS.__init__(self, rho, energy, smooth)
        self.SymSpl = UnivariateSpline(rho_Sym, Sym, s=smooth)
        self.dSymSpl = self.SymSpl.derivative(1)
        self.ddSymSpl = self.SymSpl.derivative(2)

    def GetSpeedOfSound(self, rho, pfrac):
        return (2*rho*(self.dspl(rho) + (2*pfrac - 1)**2*self.dSymSpl(rho)) 
               + rho*rho*(self.ddspl(rho) + (2*pfrac - 1)**2*self.ddSymSpl(rho))) \
               /(self.GetEnergy(rho, pfrac) 
               + (self.dspl(rho) + (2*pfrac - 1)**2*self.dSymSpl(rho))*rho)

    def GetEnergy(self, rho, pfrac):
        return SplineEOS.GetEnergy(self, rho, pfrac) + (2*pfrac - 1)**2*self.SymSpl(rho)

    def GetPressure(self, rho, pfrac):
        return SplineEOS.GetPressure(self, rho, pfrac) + rho*rho*(2*pfrac - 1)**2*self.dSymSpl(rho)

    def GetEnergyDensity(self, rho, pfrac):
        return rho*self.GetEnergy(rho, pfrac)

    def Get2EGrad(self, rho, pfrac):
        return SplineEOS.ddspl(rho) + (2*pfrac - 1)**2*self.ddSymSpl(rho)

    def GetAsymEnergy(self, rho, *args):
        return self.SymSpl(rho)

    def GetL(self, rho):
        grad_S = self.dSymSpl(rho)
        return 3*rho*grad_S



"""
Create a spline according to what you've provided. Bare minimum: energy as a function of rho
"""
def EOSSpline(rho, energy=None, pressure=None, energy_density=None, rho_Sym=None, Sym=None, smooth=0):

    if energy is None:
        energy = energy_density/rho

    if rho_Sym is None:
        rho_Sym = rho

    if Sym is None:
        eos = SplineEOS(rho, energy, smooth)
    else:
        eos = SplineEOSFull(rho, energy, rho_Sym, Sym, smooth)

    if pressure is not None:
        eos.SplPressure = UnivariateSpline(rho, pressure, s=smooth)
        def CustomGetPressure(self, rho, pfrac):
            return self.SplPressure(rho)
        eos.GetPressure = types.MethodType(CustomGetPressure, eos)

    if energy_density is not None:
        # calculation of speed of sound with energy density is more accurate
        # Especially when density is low (crustal EoS)
        eos.sound = UnivariateSpline(energy_density, pressure, s=smooth).derivative(1)
        eos.density_spl = UnivariateSpline(rho, energy_density, s=smooth)
        def CustomGetEnergyDensity(self, rho, pfrac):
            return self.density_spl(rho)
        def CustomGetSpeedOfSound(self, rho, pfrac):
            return eos.sound(rho)
        eos.GetEnergyDensity = types.MethodType(CustomGetEnergyDensity, eos)
        eos.GetSpeedOfSound = types.MethodType(CustomGetSpeedOfSound, eos)
    return eos
    


class EOSConnect(EOS):
    """
    Connect EOS from different intervals to make a single EOS
    """

    def __init__(self, intervals, eos_list):
        EOS.__init__(self)
        self.eos_list = eos_list
        self.intervals = intervals

    def GetEnergy(self, rho, pfrac):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetEnergy(rho, pfrac))(eos) for eos in self.eos_list])

    def GetEnergyDensity(self, rho, pfrac):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetEnergyDensity(rho, pfrac))(eos) for eos in self.eos_list])

    def GetPressure(self, rho, pfrac):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetPressure(rho, pfrac))(eos) for eos in self.eos_list])

    def Get2EGrad(self, rho, pfrac):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.Get2EGrad(rho, pfrac))(eos) for eos in self.eos_list])

    def GetSpeedOfSound(self, rho, pfrac):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetSpeedOfSound(rho, pfrac))(eos) for eos in self.eos_list])

    def _Interval(self, rho):
        return [(rho > interval[0]) & (rho <= interval[1]) for interval in self.intervals]


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

    def GetPressure(self, rho, pfrac):
        return self.C + self.K*np.power(self.GetEnergyDensity(rho, pfrac), 4./3.)

    def GetEnergy(self, rho, pfrac):
        return self.GetEnergyDensity(rho, pfrac)/rho

    def GetEnergyDensity(self, rho, pfrac):
        return self.A*rho + self.B

    def GetSpeedOfSound(self, rho, pfrac):
        return 4./3.*self.K*np.power(self.GetEnergyDensity(rho, 0), 1./3.)

    def Get2EGrad(self, rho, pfrac):
        pass

def SmoothPseudo(ini_rho, ini_eos, final_rho, final_eos):
    ini_energy_density = ini_eos.GetEnergyDensity(ini_rho, 0)
    ini_pressure = ini_eos.GetPressure(ini_rho, 0)
    final_energy_density = final_eos.GetEnergyDensity(final_rho, 0)
    final_pressure = final_eos.GetPressure(final_rho, 0)

    peos = PseudoEOS(ini_rho, ini_energy_density, ini_pressure, final_rho, final_energy_density, final_pressure)
    rho_range = final_rho - ini_rho
    start_rho = np.linspace(ini_rho, ini_rho + 0.1*rho_range, 10, endpoint=True)
    mid_rho = np.linspace(ini_rho + 0.3*rho_range, ini_rho + 0.4*rho_range, 0, endpoint=True)
    end_rho = np.linspace(final_rho, final_rho + 0.1*rho_range, 10, endpoint=True)
    energy = peos.GetEnergy(mid_rho, 0)
    pressure = peos.GetPressure(mid_rho, 0)

    energy = np.concatenate([ini_eos.GetEnergy(start_rho, 0), energy, final_eos.GetEnergy(end_rho, 0)])
    pressure = np.concatenate([ini_eos.GetPressure(start_rho,  0), pressure, final_eos.GetPressure(end_rho, 0)])

    eos = EOSSpline(np.concatenate([start_rho, mid_rho, end_rho]), energy=energy, pressure=pressure)

    def CustomGetSpeedOfSound(self, rho, pfrac):
        return final_eos.GetSpeedOfSound(rho, pfrac)
    eos.GetSpeedOfSound = types.MethodType(CustomGetSpeedOfSound, eos)

    return eos


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
        print(self.coeff)


    def GetEnergy(self, rho, pfrac):
        return self.coeff[0] + self.coeff[1]*rho + self.coeff[2]*np.power(rho, 2) + self.coeff[3]*np.power(rho, 3) + self.coeff[4]*np.power(rho, 4) + self.coeff[5]*np.power(rho, 5)


    

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

    def GetEnergy(self, rho, pfrac):
        return self.init_energy + self.K*(np.power(rho, self.gamma-1) - np.power(self.init_density, self.gamma-1))/(self.gamma - 1)
    

class ConstSpeed(EOS):
    
    def __init__(self, init_density, init_energy, init_pressure, speed_of_sound=0.95):
        EOS.__init__(self)
        self.init_pressure = init_pressure
        self.init_energy = init_energy
        self.init_density = init_density
        self.speed_of_sound = speed_of_sound
        self.C1 = (init_pressure + init_energy*init_density)/((speed_of_sound + 1)*np.power(init_density, speed_of_sound + 1))
        self.C2 = init_pressure - speed_of_sound*init_energy*init_density

    def GetEnergy(self, rho, pfrac):
        return self.C1*np.power(rho, self.speed_of_sound)  - self.C2/((self.speed_of_sound + 1)*rho)


class FermiGas(EOS):

  
    def __init__(self, mass):
        EOS.__init__(self)
        self.mass = mass

    def GetEnergyDensity(self, rho, pfrac):
        a = self.mass*self.mass
        b = hbar*hbar

        """
        reduce to a form of integrate x^2sqrt(a + b*x^2) dx from 0 to 3.094rho
        """
        def anti_deriv(x):
            return (np.sqrt(b*(a+b*x*x))*x*(a+2*b*x*x)-a*a*np.log(np.sqrt(b*(a+b*x*x))+b*x))/(8*np.power(b, 1.5)*pi2)

        return (anti_deriv(3.094*np.power(rho, 0.33333333333333)) - anti_deriv(0))

    def GetEnergy(self, rho, pfrac):
        return self.GetEnergyDensity(rho, pfrac)/rho

class FullEOS(EOS):
   
    def __init__(self):
        EOS.__init__(self)
    """
    Full EoS
    Where symmetric energy term is provided/ can be independently calculated
    Need to override GetAsymEnergy
    if autograd is not compatible with your GetAsymEnergy
    you need to override all other listed functions as well
    """

    def GetAsymEnergy(self, rho, *args):
        pass


    def GetK(self, rho, pfrac):
        sec_grad_density = egrad(egrad(self.GetEnergy, 0), 0)(rho, pfrac)
        return 9*rho*rho*sec_grad_density
    
    def GetQ(self, rho, pfrac):
        """
        For some weird reasons, Q is also called -K' which we will use to compare
        """
        third_grad_density = egrad(egrad(egrad(self.GetEnergy, 0), 0), 0)(rho, pfrac)
        return 27*rho*rho*rho*third_grad_density

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



class Skryme(FullEOS):



    def __init__(self, para):
        FullEOS.__init__(self)
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
    def GetEffectiveMass(self, rho, pfrac):
        result = self.a*self.__GetH(5./3., pfrac) + self.b*self.__GetH(8./3., pfrac)
        result *= (mn*rho/(4*(hbar**2)))
        result += self.__GetH(5./3., pfrac)
        return 1./result
    
    def GetEnergy(self, rho, pfrac):
        result = 3.*(hbar**2.)/(10.*mn)*((3.*pi2/2.)**0.666667)*np.power(rho, 0.6667)*self.__GetH(5./3., pfrac)
        result += self.para['t0']/8.*rho*(2.*(self.para['x0']+2.)-(2.*self.para['x0']+1)*self.__GetH(2., pfrac))
        for i in xrange(1, 4):
            result += 1./48.*self.para['t3%d'%i]*(rho**(self.para['sigma%d'%i]+1.))*(2.*(self.para['x3%d'%i]+2.)-(2.*self.para['x3%d'%i]+1.)*self.__GetH(2., pfrac))
        result += 3./40.*((3.*pi2/2.)**0.666667)*np.power(rho, 5./3.)*(self.a*self.__GetH(5./3., pfrac)+self.b*self.__GetH(8./3., pfrac))
        return result + mn
    
    def GetAsymEnergy(self, rho, *args):
        result = (hbar**2.)/(6.*mn)*((3.*pi2/2.)**0.666667)*np.power(rho, 0.6667)
        result -= self.para['t0']/8.*rho*(2.*self.para['x0']+1.)
        for i in xrange(1, 4):
            result -= 1./48.*self.para['t3%d'%i]*(rho**(self.para['sigma%d'%i]+1.))*(2.*self.para['x3%d'%i]+1.)
        result += 1./24.*((3.*pi2/2.)**0.666667)*np.power(rho, 5./3.)*(self.a+4*self.b)
        return result

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


    

def SummarizeSkyrme(df):
    """
    This function will print out the value of E0, K0, K'=-Q0, J=S(rho0), L, Ksym, Qsym, m*
    """
    summary_list = []
    #print('Model\tE0\tK0\tK\'\tJ\tL\tKsym\tQsym\tm*')

    """
    Check if it is skyrme first, if not then it will return an empty dataframe
    """
    if 't0' not in df.iloc[0]:
        return pd.DataFrame()

    for index, row in df.iterrows():
        sky = Skryme(row)
        rho0 = sky.rho0
        E0 = sky.GetEnergy(rho0, 0.5)
        K0 = sky.GetK(rho0, 0.5)
        Kprime = -sky.GetQ(rho0, 0.5)
        J = sky.GetAsymEnergy(rho0)
        L = sky.GetL(rho0)
        Ksym = sky.GetKsym(rho0)
        Qsym = sky.GetQsym(rho0)
        eff_m = sky.GetEffectiveMass(rho0, 0.5)
        m_n = sky.GetEffectiveMass(rho0, 0.)
        m_p = sky.GetEffectiveMass(rho0, 1)
        #print('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (index, E0, K0, Kprime, J, L, Ksym, Qsym, eff_m))
        summary_dict = {'Model':index, 'E0':E0, 'K0':K0, 'K\'':Kprime, 'J':J, 'L':L, 'Ksym':Ksym, 'Qsym':Qsym, 'm*':eff_m, 'm_n': m_n, 'm_p': m_p}
        summary_list.append(summary_dict)

    df = pd.DataFrame.from_dict(summary_list)
    df.set_index('Model', inplace=True)
    return df
    
