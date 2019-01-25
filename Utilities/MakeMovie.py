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

def CreateGif(df_list, output_filename, densitylist, func='GetAsymEnergy', ymin=0, ymax=100, xmin=60, xmax=1500, color=['r', 'b'], xval='lambda(1.4)', xlabel=r'$Deformability\ \Lambda$', ylabel=r'$Sym (MeV/fm^{3})$', xscale='log', yscale='linear'):
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
                polarizability.append(row[xval])
            ax.plot(polarizability, pressure, 'ro', marker='o', markerfacecolor='w', color=color[num])
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([xmin,xmax])
        ax.set_xscale(xscale)
        ax.set_ylim([ymin, ymax])
        ax.text(0.1, 0.85, r'$Density = %3.2f\rho_{0}$' % density, fontsize=35, transform=ax.transAxes)
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

    CreateGif([df], 'Sym_0_1.5_R.gif', np.linspace(0.1, 1.5, 50).tolist(), xval='R(1.4)', xmin=7, xmax=16, xscale='linear', xlabel=r'Radius (km)')
    CreateGif([df], 'Pressure_0_1.5_R.gif', np.linspace(0.1, 1.5, 50).tolist(), 'GetPressure', 0, 50, xval='R(1.4)', xmin=7, xmax=16, xscale='linear', xlabel=r'Radius (km)', ylabel=r'Pressure $(MeV/fm^{3})$')

    CreateGif([df], 'Sym_1.5_2.5_R.gif', np.linspace(1.5, 2.5, 50).tolist(), xval='R(1.4)', xmin=7, xmax=16, xscale='linear', xlabel=r'Radius (km)')
    CreateGif([df], 'Pressure_1.5_2.5_R.gif', np.linspace(1.5, 2.5, 50).tolist(), 'GetPressure', 0, 50, xval='R(1.4)', xmin=7, xmax=16, xscale='linear', xlabel=r'Radius (km)', ylabel=r'Pressure $(MeV/fm^{3})$')



    CreateGif([df], 'Sym_0_1.5.gif', np.linspace(0.1, 1.5, 50).tolist())
    CreateGif([df], 'Pressure_0_1.5.gif', np.linspace(0.1, 1.5, 50).tolist(), 'GetPressure', 0, 50, ylabel=r'Pressure $(MeV/fm^{3})$')

    CreateGif([df], 'Sym_1.5_2.5.gif', np.linspace(1.5, 2.5, 50).tolist())
    CreateGif([df], 'Pressure_1.5_2.5.gif', np.linspace(1.5, 2.5, 50).tolist(), 'GetPressure', 0, 50, ylabel=r'Pressure $(MeV/fm^{3})$')




