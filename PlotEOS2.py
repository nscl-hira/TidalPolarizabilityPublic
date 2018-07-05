import sys
from pebble import ProcessPool
import scipy.optimize as opt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import Utilities.Utilities as tul
import Utilities.SkyrmeEOS as sky
from Utilities.Constants import *
from EOSCreator import EOSCreator

from itertools import izip


if __name__ == "__main__":

    df = pd.read_csv('Results/Skyrme_pd_0.7_pp_3.csv', index_col=0)
    df.fillna(0, inplace=True)

    def GetEnergyDensityVsPressure(eos_name):
        skyrme = sky.Skryme(df.loc[eos_name])
        #skyrme.ToCSV('AllSkyrmes/%s.csv' % eos_name, np.linspace(1e-14, 3*0.16, 100), 0)
        eos_creator = EOSCreator(skyrme, PRCTransDensity=0.7)
        pressure_high = df['PolyHighP'].loc[eos_name]
        eos_creator.PressureHigh = pressure_high
        eos, trans_dens = eos_creator.GetEOSType("EOS")
    
        trans_dens = [7*rho0] + trans_dens + [1e-8]  
        pressure_list = []
        energy_density_list = []
        rho_list = []

        for low_den, high_den in zip(trans_dens, trans_dens[1:]):
            rho = np.linspace(low_den, high_den, 100)
            rho_list.append(rho)
            pressure_list.append(eos.GetAutoGradPressure(rho, 0))
            energy_density_list.append(eos.GetEnergyDensity(rho, 0))
        print('finishted %s' % eos_name)
        return rho_list, pressure_list, energy_density_list

        
        

    name_list = [index for index, row in df.iterrows()]
    result = []
    with ProcessPool() as pool:
        future = pool.map(GetEnergyDensityVsPressure, name_list)
        iterator = future.result()
        while True:
            try:
                result.append(next(iterator))
            except StopIteration:
                break
            except:
                print sys.exc_info()[0]
                raise
    ax = plt.subplot(111)
    first = True
    for val in result:
        color = ['r', 'b', 'g', 'orange', 'b']
        if first:
            label = ['Polytrope', 'Skyrme', 'Rel. gas', 'Crust']
        else:
            label = [None, None, None, None]

        for index, (rho, pressure, energy_density) in enumerate(zip(val[0], val[1], val[2])):
            ax.plot(rho, pressure, color=color[index], label=label[index])
        first = False

    ax.legend()
    ax.set_ylabel(r'$Pressure (MeV/fm^{3})$')
    ax.set_xlabel(r'$Density (fm^{-3})$')
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-2, 1e3)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
            
