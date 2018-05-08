import autograd.numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
import math

from Constants import *


def GetH(n, pfrac):
    return (2**(n-1))*(pfrac**n + (1-pfrac)**n)

"""
Return m* = M*/M instead of M*
"""
def GetEffectiveMass(rho, pfrac, para):
    a = para['t1']*(para['x1']+2) + para['t2']*(para['x2']+2)
    b = 0.5*(para['t2']*(2*para['x2']+1)-para['t1']*(2*para['x1']+1))

    result = a*GetH(5./3., pfrac) + b*GetH(8./3., pfrac)
    result *= (mn*rho/(4*(hbar**2)))
    result += GetH(5./3., pfrac)
    return 1./result

def GetEnergy(rho, pfrac, para):
    a = para['t1']*(para['x1']+2) + para['t2']*(para['x2']+2)
    b = 0.5*(para['t2']*(2*para['x2']+1)-para['t1']*(2*para['x1']+1))

    result = 3.*(hbar**2.)/(10.*mn)*((3.*pi2/2.)**0.666667)*np.power(rho, 0.6667)*GetH(5./3., pfrac)
    result += para['t0']/8.*rho*(2.*(para['x0']+2.)-(2.*para['x0']+1)*GetH(2., pfrac))
    for i in xrange(1, 4):
        result += 1./48.*para['t3%d'%i]*(rho**(para['sigma%d'%i]+1.))*(2.*(para['x3%d'%i]+2.)-(2.*para['x3%d'%i]+1.)*GetH(2., pfrac))
    result += 3./40.*((3.*pi2/2.)**0.666667)*np.power(rho, 5./3.)*(a*GetH(5./3., pfrac)+b*GetH(8./3., pfrac))
    return result

def GetAutoGradPressure(rho, pfrac, para):
    grad_edensity = egrad(GetEnergy, 0)(rho, pfrac, para)
    return rho*rho*grad_edensity

def GetAutoGradK(rho, pfrac, para):
    sec_grad_density = egrad(egrad(GetEnergy, 0), 0)(rho, pfrac, para)
    return 9*rho*rho*sec_grad_density

def GetAutoGradQ(rho, pfrac, para):
    """
    For some weird reasons, Q is also called -K' which we will use to compare
    """
    third_grad_density = egrad(egrad(egrad(GetEnergy, 0), 0), 0)(rho, pfrac, para)
    return 27*rho*rho*rho*third_grad_density

def GetAsymEnergy(rho, para):
    a = para['t1']*(para['x1']+2) + para['t2']*(para['x2']+2)
    b = 0.5*(para['t2']*(2*para['x2']+1)-para['t1']*(2*para['x1']+1))

    result = (hbar**2.)/(6.*mn)*((3.*pi2/2.)**0.666667)*np.power(rho, 0.6667)
    result -= para['t0']/8.*rho*(2.*para['x0']+1.)
    for i in xrange(1, 4):
        result -= 1./48.*para['t3%d'%i]*(rho**(para['sigma%d'%i]+1.))*(2.*para['x3%d'%i]+1.)
    result += 1./24.*((3.*pi2/2.)**0.666667)*np.power(rho, 5./3.)*(a+4*b)
    return result

"""
Usually these functions are defined only at rho0
When calling them, try to set rho to rho0
"""
def GetL(rho, para):
    grad_S = egrad(GetAsymEnergy, 0)(rho, para)
    return 3*rho*grad_S

def GetKsym(rho, para):
    second_grad_S = egrad(egrad(GetAsymEnergy, 0), 0)(rho, para)
    return 9*rho*rho*second_grad_S

def GetQsym(rho, para):
    third_grad_S = egrad(egrad(egrad(GetAsymEnergy, 0), 0), 0)(rho, para)
    return 27*rho*rho*rho*third_grad_S


def SummarizeSkyrme(df):
    """
    This function will print out the value of E0, K0, K'=-Q0, J=S(rho0), L, Ksym, Qsym, m*
    """
    summary_list = []
    print('Model\tE0\tK0\tK\'\tJ\tL\tKsym\tQsym\tm*')
    for index, row in df.iterrows():
        E0 = GetEnergy(rho0, 0.5, row)
        K0 = GetAutoGradK(rho0, 0.5, row)
        Kprime = -GetAutoGradQ(rho0, 0.5, row)
        J = GetAsymEnergy(rho0, row)
        L = GetL(rho0, row)
        Ksym = GetKsym(rho0, row)
        Qsym = GetQsym(rho0, row)
        eff_m = GetEffectiveMass(rho0, 0.5, row)
        print('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (index, E0, K0, Kprime, J, L, Ksym, Qsym, eff_m))
        summary_dict = {'Model':index, 'E0':E0, 'K0':K0, 'K\'':Kprime, 'J':J, 'L':L, 'Ksym':Ksym, 'Qsym':Qsym, 'm*':eff_m}
        summary_list.append(summary_dict)

    df = pd.DataFrame.from_dict(summary_list)
    df.set_index('Model', inplace=True)
    return df
    
