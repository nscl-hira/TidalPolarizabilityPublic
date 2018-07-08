import autograd.numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
import math
import scipy.misc as misc
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from Constants import *

class PseudoEOS:

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

    def GetAutoGradPressure(self, rho, pfrac):
        return self.C + self.K*np.power(self.GetEnergyDensity(rho, pfrac), 4./3.)

    def GetEnergy(self, rho, pfrac):
        return self.GetEnergyDensity(rho, pfrac)/rho

    def GetEnergyDensity(self, rho, pfrac):
        return self.A*rho + self.B

    def GetSpeedOfSound(self, rho, pfrac):
        return 4./3.*self.K*np.power(self.GetEnergyDensity(rho, 0), 1./3.)

class EOSSpline:


    def __init__(self, rho, energy=None, smooth=0, rho_Sym=None, Sym=None, smooth_sym=0.5, pressure=None, energy_density=None):
        if energy_density is None:
            self.spl = UnivariateSpline(rho, energy)#, s=smooth)
            
            self.density_spl = None
        else:
            self.spl = UnivariateSpline(rho, energy_density/rho, s=smooth)
            self.density_spl = UnivariateSpline(rho, energy_density, s=smooth)
            
        self.dspl = self.spl.derivative(1)
        self.ddspl = self.spl.derivative(2)
        
        if rho_Sym is None:
            rho_Sym = rho
        if Sym is None:
            self.SymSpl = lambda rho: 0
            self.dSymSpl = self.SymSpl
            self.ddSymSpl = self.SymSpl
        else:
            self.SymSpl = UnivariateSpline(rho_Sym, Sym, s=smooth_sym)
            self.dSymSpl = self.SymSpl.derivative(1)
            self.ddSymSpl = self.SymSpl.derivative(2)

        self.sound = None
        if pressure is None:
            self.SplPressure = None
        else:
            self.SplPressure = UnivariateSpline(rho, pressure, s=0)
            if energy_density is not None:
                self.sound = UnivariateSpline(energy_density, pressure, s=smooth)
        #plt.plot(rho, energy, 'ro')
        #plt.plot(0.16*np.linspace(0.1, 3, 100), self.spl(0.16*np.linspace(0.1, 3, 100)))
        #plt.plot(0.16*np.linspace(0.1, 3, 100), self.dSymSpl(0.16*np.linspace(0.1, 3, 100)))
        #plt.show()


    def GetEnergy(self, rho, pfrac):
        return self.spl(rho) + (2*pfrac - 1)**2*self.SymSpl(rho)

    def GetEnergyDensity(self, rho, pfrac):
        if self.density_spl is None:
            return rho*self.GetEnergy(rho, pfrac)
        else:
            return self.density_spl(rho)

    def GetAutoGradPressure(self, rho, pfrac):
        if self.SplPressure is None:
            grad_edensity = self.dspl(rho) + (2*pfrac - 1)**2*self.dSymSpl(rho)
            pressure = rho*rho*grad_edensity
        else:
            pressure = self.SplPressure(rho)
        return pressure

    def GetSpeedOfSound(self, rho, pfrac):
        if self.sound is not None:
            return self.sound.derivative(1)(rho)
        else:
            return (2*rho*(self.dspl(rho) + (2*pfrac - 1)**2*self.dSymSpl(rho)) 
                   + rho*rho*(self.ddspl(rho) + (2*pfrac - 1)**2*self.ddSymSpl(rho))) \
                   /(self.GetEnergy(rho, pfrac) 
                   + (self.dspl(rho) + (2*pfrac - 1)**2*self.dSymSpl(rho))*rho)

    def GetAsymEnergy(self, rho):
        return self.SymSpl(rho)
    

class EOS:

    def __init__(self):
        pass

    """
    For the program to work, just program the following 3 entries:
    GetEnergy
    GetEffectiveMass
    GetAsymEnergy

    Gradient will be calculated with autograd, so try to write the function in an explicity mannar
    Avoid unsupported library
    """
    def GetEnergy(self, rho, pfrac):
        pass

    def GetEnergyDensity(self, rho, pfrac):
        return rho*self.GetEnergy(rho, pfrac)

    def GetEffectiveMass(self, rho, pfrac):
        pass

    def GetAsymEnergy(self, rho):
        pass

    def GetAutoGradPressure(self, rho, pfrac):
        grad_edensity = egrad(self.GetEnergy, 0)(rho, pfrac)
        return rho*rho*grad_edensity

    def GetAutoGradK(self, rho, pfrac):
        sec_grad_density = egrad(egrad(self.GetEnergy, 0), 0)(rho, pfrac)
        return 9*rho*rho*sec_grad_density
    
    def GetAutoGradQ(self, rho, pfrac):
        """
        For some weird reasons, Q is also called -K' which we will use to compare
        """
        third_grad_density = egrad(egrad(egrad(self.GetEnergy, 0), 0), 0)(rho, pfrac)
        return 27*rho*rho*rho*third_grad_density

    def GetSpeedOfSound(self, rho, pfrac):
        return egrad(self.GetAutoGradPressure, 0)(rho, pfrac)/(self.GetEnergy(rho, pfrac) + egrad(self.GetEnergy, 0)(rho, pfrac)*rho)
    
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
            
            


class PolyTrope(EOS):


    def __init__(self, init_density, init_energy, init_pressure, final_density, final_pressure, gamma=None):
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
    
    """
    Polytrope is not a full equation
    It doesn't handle proton fraction at all
    """
    def GetEffectiveMass(self, rho, pfrac):
        return 0

    def GetAsymEnergy(self, rho):
        return 0

class ConstSpeed(EOS):
    
    def __init__(self, init_density, init_energy, init_pressure, speed_of_sound=0.95):
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

    """
    Fermi gas is not a full equation
    It doesn't handle proton fraction at all
    """
    def GetEffectiveMass(self, rho, pfrac):
        return 0

    def GetAsymEnergy(self, rho):
        return 0



class Skryme(EOS):



    def __init__(self, para):
        self.para = para
        self.a = para['t1']*(para['x1']+2) + para['t2']*(para['x2']+2)
        self.b = 0.5*(para['t2']*(2*para['x2']+1)-para['t1']*(2*para['x1']+1))

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
    
    def GetAsymEnergy(self, rho):
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
        pressure = self.GetAutoGradPressure(rho, pfrac)
        data = np.vstack((sym_energy, energy, energy_density, pressure, rho, rho/rho0)).T
       
        with open(filename, 'w') as file_:
            file_.write('''({dashes}
{S:^12},{EV:^12},{EN:^12},{P:^12},{rho:^12},{rho_rho0}:^12
{dashes})\n'''.format(dashes='-'*106, S='S(rho) (MeV)', EV='E/V (MeV/fm3)', EN='E/N MeV', P='P (MeV/fm3)', rho='rho (fm-3)', rho_rho0='rho/rho0'))
            np.savetxt(file_, data, fmt='%12.3f', delimiter=',')


class EOSConnect(EOS):


    def __init__(self, intervals, eos_list):
        self.eos_list = eos_list
        self.intervals = intervals

    def GetEnergy(self, rho, pfrac):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetEnergy(rho, pfrac))(eos) for eos in self.eos_list])

    def GetEnergyDensity(self, rho, pfrac):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetEnergyDensity(rho, pfrac))(eos) for eos in self.eos_list])

    def GetAutoGradPressure(self, rho, pfrac):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetAutoGradPressure(rho, pfrac))(eos) for eos in self.eos_list])

    def GetSpeedOfSound(self, rho, pfrac):
        return np.piecewise(rho, self._Interval(rho), [(lambda func: lambda rho: func.GetSpeedOfSound(rho, pfrac))(eos) for eos in self.eos_list])

    def _Interval(self, rho):
        return [(rho > interval[0]) & (rho <= interval[1]) for interval in self.intervals]
    

def SummarizeSkyrme(df):
    """
    This function will print out the value of E0, K0, K'=-Q0, J=S(rho0), L, Ksym, Qsym, m*
    """
    summary_list = []
    #print('Model\tE0\tK0\tK\'\tJ\tL\tKsym\tQsym\tm*')
    for index, row in df.iterrows():
        sky = Skryme(row)
        E0 = sky.GetEnergy(rho0, 0.5)
        K0 = sky.GetAutoGradK(rho0, 0.5)
        Kprime = -sky.GetAutoGradQ(rho0, 0.5)
        J = sky.GetAsymEnergy(rho0)
        L = sky.GetL(rho0)
        Ksym = sky.GetKsym(rho0)
        Qsym = sky.GetQsym(rho0)
        eff_m = sky.GetEffectiveMass(rho0, 0.5)
        #print('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (index, E0, K0, Kprime, J, L, Ksym, Qsym, eff_m))
        summary_dict = {'Model':index, 'E0':E0, 'K0':K0, 'K\'':Kprime, 'J':J, 'L':L, 'Ksym':Ksym, 'Qsym':Qsym, 'm*':eff_m}
        summary_list.append(summary_dict)

    df = pd.DataFrame.from_dict(summary_list)
    df.set_index('Model', inplace=True)
    return df
    
