from autograd import elementwise_grad as egrad
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import scipy.optimize as optimize
import pandas as pd

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *


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

    rho = np.linspace(0.1,10,100)
    min_result = [optimize.minimize(lambda frac: GetEnergy(rho_*rho0, frac[0], frac[1]), [0.5, 0.5], bounds=[(1e-3, 1), (1e-3,1)], method='SLSQP', options={'disp':False}) for rho_ in rho]
    pfrac = [min_.x[0] for min_ in min_result]
    mufrac = [min_.x[1]*min_.x[0] for min_ in min_result]
    plt.plot(rho, pfrac, label='proton fraction')
    plt.plot(rho, mufrac, label='muon fraction')
    plt.legend()
    plt.show()

    energy = [min_.fun for min_ in min_result]
    plt.plot(rho, GetEnergy(rho*rho0, 0.5, 0), label='Sym matter')
    plt.plot(rho, GetEnergy(rho*rho0, 0.0, 0), label='Pure neutron matter')
    plt.plot(rho, energy, label='Equabilium')
    plt.legend()
    plt.show()
