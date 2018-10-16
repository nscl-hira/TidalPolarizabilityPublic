from decimal import Decimal
from TidalLove import TidalLove_CPP as tidal
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import pandas as pd
import tempfile
import math

from MakeSkyrmeFileBisection import LoadSkyrmeFile
import Utilities as ult
from Utilities.EOSCreator import EOSCreator

def PressureComposition(ax, eos_name, filename):
    df = LoadSkyrmeFile(filename)
    row = df.loc[eos_name]
    creator = EOSCreator(row)#SkyrmeDensity = 1, TranDensity=1, CrustSmooth=0, PRCTransDensity=-1, **row)
    kwargs = creator.PrepareEOS(**row)    

    eos, trans_dens = creator.GetEOSType(**row)
    
    rho0 = 0.16

    trans_dens = [10*rho0] + trans_dens + [1e-9]
    trans_dens = np.array(trans_dens)
    trans_pressure = eos.GetPressure(trans_dens, 0)

    file_ = tempfile.NamedTemporaryFile()
    file_.write("asfd\nasdf\nasdf\nasdf\n")
    n = np.concatenate([np.logspace(np.log(1e-10), np.log(3.76e-4), 2000, base=np.exp(1)), np.linspace(3.77e-4, 10, 18000)])
    #n = np.linspace(1e-12, 10, 20000) 
    energy = (eos.GetEnergyDensity(n, 0.))
    pressure = eos.GetPressure(n, 0.) 
    for density, e, p in zip(n, energy, pressure):
        if(not math.isnan(e) and not math.isnan(p)):
            file_.write("   %.5e   %.5e   %.5e   0.0000e+0\n" % (Decimal(e), Decimal(p), Decimal(density)))
    file_.flush()
    
    mass, radius, pressure = tidal.tidallove_analysis(file_.name, 81.)
    data = pd.DataFrame.from_dict({'mass':mass, 'radius':radius, 'pressure':pressure})
    color = ['r', 'b', 'g', 'orange', 'b', 'pink']
    labels = ['', 'Crustal EOS', 'Electron gas', 'Skyrme'] + ['']*(len(trans_dens) - 4)
    data = data[data['pressure'] > 1e-8]
    for num, trans_den in enumerate(trans_pressure):
        data_subrange = data[data['pressure'] < trans_den]
        radius = data_subrange['radius']
        pressure = data_subrange['pressure']
        mass = data_subrange['mass']
        ax.plot(radius, pressure, color=color[num], label=labels[-num-1])
    ax.legend()
    ax.set_xlabel('r (km)')
    ax.set_ylabel('P (MeV/c)')
    
if __name__ == '__main__':
    fig, ax = plt.subplots()
    PressureComposition(ax, 'SLy8', 'Results/Newest.csv')
    #PressureComposition('MSkA', 'Results/Newest_AACrust.csv')
    #plt.yscale('log')
    plt.show()
