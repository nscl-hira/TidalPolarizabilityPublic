from autograd import elementwise_grad as egrad
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import scipy.optimize as optimize
import pandas as pd

from scipy.interpolate import UnivariateSpline
import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from BetaEquilibrium import BetaEquilibrium
from EOSCreator import EOSCreator


if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_pd_0.7_pp_2.5_mm_2.5.csv', index_col=0)
    df.fillna(0, inplace=True)
 
    #Nuclear = sky.Skryme(df.loc['BSk1'])
    #Crust, _ = EOSCreator(Nuclear, SkyrmeDensity=0.3*rho0, CrustSmooth=0).GetOnlySkyrme() #GetCrust()
    rho = np.linspace(1e-10,rho0*3,100)
    #rho_crust = np.logspace(np.log(1e-12), np.log(0.045), 1000, base=np.exp(1))

    first = True
    for key, eos in df.iterrows():
        Nuclear = sky.Skryme(eos)
        if first:
            label = 'Skyrme'
            first = False
        else:
            label = None
        plt.plot(Nuclear.GetAutoGradPressure(rho, 0), Nuclear.GetEnergyDensity(rho, 0), label=label, color='r')
        Nuclear.ToCSV('AllSkyrmes/NoNeutronStar/%s.csv' % key, rho, 0)
    #plt.plot(Crust.GetAutoGradPressure(rho, 0), Crust.GetEnergyDensity(rho, 0), label='Crust', color='b')
    plt.ylabel(r'$Energy Density (MeV/fm^{3})$')
    plt.xlabel(r'$Pressure (MeV/fm^{3})$')
    """
    plt.ylim([1e-14, 1e4])
    plt.xlim([1e-14, 1e5])
    plt.yscale('log')
    plt.xscale('log')
    """
    plt.legend()
    plt.show()
    """
    lns = sky.Skryme(df.loc['LNS'])
    plt.plot(Crust.GetAutoGradPressure(rho_crust, 0), np.sqrt(Crust.GetSpeedOfSound(rho_crust, 0)))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Speed of sound / c')
    plt.xlabel(r'$Pressure (MeV/fm^{3})$')
    plt.show()
    """
    """
    index = 0
    rho = np.logspace(np.log(1e-5),np.log(10),500)*rho0
    first = True
    for key, eos in df.iterrows():
        Nuclear, ele, mu_ = BetaEquilibrium(sky.Skryme(eos))
        if first:
            label = 'Skyrme'
            first = False
        else:
            label = None
        plt.plot(rho, Nuclear.GetAutoGradPressure(rho, 0), color='r')
        plt.plot(rho, ele.GetAutoGradPressure(rho, 0), color='g')
        plt.plot(rho, mu_.GetAutoGradPressure(rho, 0), color='b')
        index = index + 1
        if index > 50:
            break
    plt.xlabel(r'$Density (fm^{-3})$')
    plt.ylabel(r'$Pressure (MeV/fm^{3})$')
    plt.ylim([1e-4, 1e4])
    plt.xlim([1e-5, 1])
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()

    index = 0
    rho = np.logspace(np.log(1e-5),np.log(10),500)*rho0
    first = True
    for key, eos in df.iterrows():
        Nuclear, ele, mu_ = BetaEquilibrium(sky.Skryme(eos))
        if first:
            label = 'Skyrme'
            first = False
        else:
            label = None
        plt.plot(rho, Nuclear.GetEnergyDensity(rho, 0), color='r')
        plt.plot(rho, ele.GetEnergyDensity(rho, 0), color='g')
        plt.plot(rho, mu_.GetEnergyDensity(rho, 0), color='b')
        index = index + 1
        if index > 50:
            break
    plt.plot(rho_crust, Crust.GetEnergyDensity(rho_crust, 0), label='Crust', color='black')
    plt.xlabel(r'$Density (fm^{-3})$')
    plt.ylabel(r'$Energy Density (MeV/fm^{3})$')
    plt.ylim([1e-2, 1e5])
    plt.xlim([1e-5, 1])
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()
    """
