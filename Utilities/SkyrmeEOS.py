import autograd.numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
import math

from Constants import *

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
    def GetEnergy(rho, pfrac):
        pass

    def GetEffectiveMass(rho, pfrac):
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
        return result
    
    def GetAsymEnergy(self, rho):
        result = (hbar**2.)/(6.*mn)*((3.*pi2/2.)**0.666667)*np.power(rho, 0.6667)
        result -= self.para['t0']/8.*rho*(2.*self.para['x0']+1.)
        for i in xrange(1, 4):
            result -= 1./48.*self.para['t3%d'%i]*(rho**(self.para['sigma%d'%i]+1.))*(2.*self.para['x3%d'%i]+1.)
        result += 1./24.*((3.*pi2/2.)**0.666667)*np.power(rho, 5./3.)*(self.a+4*self.b)
        return result
    

def SummarizeSkyrme(df):
    """
    This function will print out the value of E0, K0, K'=-Q0, J=S(rho0), L, Ksym, Qsym, m*
    """
    summary_list = []
    print('Model\tE0\tK0\tK\'\tJ\tL\tKsym\tQsym\tm*')
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
        print('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (index, E0, K0, Kprime, J, L, Ksym, Qsym, eff_m))
        summary_dict = {'Model':index, 'E0':E0, 'K0':K0, 'K\'':Kprime, 'J':J, 'L':L, 'Ksym':Ksym, 'Qsym':Qsym, 'm*':eff_m}
        summary_list.append(summary_dict)

    df = pd.DataFrame.from_dict(summary_list)
    df.set_index('Model', inplace=True)
    return df
    
