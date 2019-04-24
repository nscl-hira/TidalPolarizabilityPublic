from autograd import elementwise_grad as egrad
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import scipy.optimize as optimize
import pandas as pd

import Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *


def BetaEquilibrium(SkyrmeEOS, rho=np.concatenate([np.linspace(1e-10, 0.09, 100), np.linspace(0.1,10,100)])):
    """
    This function will return a equilibrated Skyrme in terms of EOSSpline
    from 0.1rho0 to 3rho0
    """

    ele_fermi = sky.FermiGas(me)
    mu_fermi = sky.FermiGas(mmu)

    def GetEnergy(rho, pfrac, mufrac):
        nuc_energy = SkyrmeEOS.GetEnergyDensity(rho, pfrac)
        ele_energy = ele_fermi.GetEnergyDensity(rho*pfrac*(1-mufrac), 0)
        mu_energy = mu_fermi.GetEnergyDensity(rho*pfrac*mufrac, 0)
        return nuc_energy + ele_energy + mu_energy

    rho0 = SkyrmeEOS.rho0
    min_result = [optimize.minimize(lambda frac: GetEnergy(rho_*rho0, frac[0], frac[1]), [0.5, 0.5], bounds=[(1e-14, 1), (1e-14,1)], method='SLSQP', options={'disp':False}) for rho_ in rho]

    energy = [min_.fun for min_ in min_result]
    pfrac = np.array([min_.x[0] for min_ in min_result])
    mufrac = np.array([min_.x[1]*min_.x[0] for min_ in min_result])
    return sky.SplineEOS.Construct(rho*rho0, energy/(rho*rho0), smooth=0.1), rho*rho0, pfrac, mufrac


if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_summary.csv', index_col=0)
    df.fillna(0, inplace=True)

    ele_fermi = sky.FermiGas(me)
    mu_fermi = sky.FermiGas(mmu)
    Nuclear = sky.Skryme(df.loc['LNS'])
    print(sky.FermiGas(mn).GetEnergy(0.16, 0) - mn)

    def GetEnergy(rho, pfrac, mufrac):
        nuc_energy = Nuclear.GetEnergyDensity(rho, pfrac)
        ele_energy = ele_fermi.GetEnergyDensity(rho*pfrac*(1-mufrac), 0)
        mu_energy = mu_fermi.GetEnergyDensity(rho*pfrac*mufrac, 0)
        return nuc_energy + ele_energy + mu_energy

    def GetPressure(rho, pfrac, mufrac):
        nuc_pressure = Nuclear.GetPressure(rho, pfrac)
        ele_pressure = ele_fermi.GetPressure(rho*pfrac*(1-mufrac), 0)
        mu_pressure = mu_fermi.GetPressure(rho*pfrac*mufrac, 0)
        return nuc_pressure + ele_pressure + mu_pressure

    rho = np.concatenate([np.linspace(1e-10, 0.099, 100), np.linspace(0.1,10,100)])
    min_result = [optimize.minimize(lambda frac: GetEnergy(rho_*rho0, frac[0], frac[1]), [0.5, 0.5], bounds=[(1e-4, 1), (1e-4,1)], method='SLSQP', options={'disp':False}) for rho_ in rho]
    pfrac = [min_.x[0] for min_ in min_result]
    mufrac = [min_.x[1]*min_.x[0] for min_ in min_result]
    
    plt.plot(rho, pfrac, label='proton')
    plt.plot(rho, mufrac, label='muon')
    plt.ylabel(r'Particle fraction')
    plt.xlabel(r'$\rho/\rho_{0}$')
    plt.legend()
    plt.show()

    energy = [min_.fun for min_ in min_result]
    eos_spline = sky.EOSSpline(rho*rho0, energy/(rho*rho0))
    plt.plot(rho, GetEnergy(rho*rho0, 0.5, 1e-5), label='Sym matter')
    plt.plot(rho, GetEnergy(rho*rho0, 1e-5, 1e-5), label='Pure neutron matter', linewidth=7)
    plt.plot(rho, energy, label='Equabilium', linewidth=5)
    plt.plot(rho, eos_spline.GetEnergyDensity(rho*rho0, 0), label='spline')
    plt.ylabel(r'$Energy density (MeV/fm^{3})$')
    plt.xlabel(r'$\rho/\rho_{0}$')
    plt.legend()
    plt.show()
    
    plt.plot(rho, GetPressure(rho*rho0, 0.5, 1e-5), label='Sym matter')
    plt.plot(rho, GetPressure(rho*rho0, 1e-5, 1e-5), label='Pure neutron matter')
    plt.plot(rho, GetPressure(rho*rho0, np.array(pfrac), np.array(mufrac)), label='Equilibrum', linewidth=5)
    plt.plot(rho, eos_spline.GetPressure(rho*rho0, 0), label='Spline')
    plt.ylabel(r'$Pressure (MeV/fm^{3})$')
    plt.xlabel(r'$\rho/\rho_{0}$')
    plt.legend()
    plt.show()
