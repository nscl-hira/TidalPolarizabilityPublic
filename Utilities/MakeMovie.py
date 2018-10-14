import itertools
color = itertools.cycle(('g', 'purple', 'r', 'black', 'orange')) 
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import pandas as pd
import imageio
import tempfile

import Utilities as utl
import SkyrmeEOS as sky 
from Constants import *

def CreateGif(df_list, output_filename, densitylist, func='GetAsymEnergy', ymin=0, ymax=100, color=['r', 'b']):
    # plot with different densities
    if func == 'GetAsymEnergy':
        ytitle = 'Sym'
    else:
        ytitle = 'P'
    rho0 = 0.16
    image_name = []
    for density in densitylist:#np.linspace(0.1, 2.5, 100).tolist():
        fig, ax = plt.subplots(1, 1)
        pressure = []
        polarizability = []
        for num, df in enumerate(df_list):
            for index, row in df.iterrows():
                eos = sky.Skryme(row)
                pressure.append(getattr(eos, func)(density*rho0, 0))
                polarizability.append(row['lambda(1.4)'])
            ax.plot(polarizability, pressure, 'ro', marker='o', markerfacecolor='w', color=color[num])
        ax.set_ylim([1e-2,3000])
        #ax.set_yscale('log')
        ax.set_xlabel(r'$Deformability\ \Lambda$')
        ax.set_ylabel(r'$%s (MeV/fm^{3})$' % ytitle)
        ax.set_xlim([60,1500])
        ax.set_xscale('log')
        ax.set_ylim([ymin, ymax])
        ax.text(100, 0.85*ymax, r'$Density = %3.2f\rho_{0}$' % density, fontsize=35)
        #plt.show()
        name = 'images/test%f.png' % density
        fig.savefig(name, dpi=50)
        image_name.append(name)
        plt.close(fig)

    images = []
    for filename in image_name:
        images.append(imageio.imread(filename))
    imageio.mimsave(output_filename, images, loop=1)

if __name__ == "__main__":
    df = pd.read_csv('Results/Newest.csv', index_col=0)
    df.fillna(0, inplace=True)

    CreateGif([df], 'Sym_0_1.5.gif', np.linspace(0.1, 1.5, 50).tolist())
    CreateGif([df], 'Pressure_0_1.5.gif', np.linspace(0.1, 1.5, 50).tolist(), 'GetPressure', 0, 50)

    CreateGif([df], 'Sym_1.5_2.5.gif', np.linspace(1.5, 2.5, 50).tolist())
    CreateGif([df], 'Pressure_1.5_2.5.gif', np.linspace(1.5, 2.5, 50).tolist(), 'GetPressure', 0, 50)


