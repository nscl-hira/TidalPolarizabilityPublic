import itertools
color = itertools.cycle(('g', 'purple', 'r', 'black', 'orange')) 
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import pandas as pd
import imageio
import tempfile

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *

def CreateGif(df_list, output_filename, func='GetAsymEnergy', ymin=0, ymax=100, color=['r', 'b']):
    # plot with different densities
    if func == 'GetAsymEnergy':
        ytitle = 'Sym'
    else:
        ytitle = 'P'

    image_name = []
    for density in np.linspace(0.1, 2.5, 100).tolist():
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
    imageio.mimsave(output_filename, images)

if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_pd_0.7.csv', index_col=0)
    df.fillna(0, inplace=True)

    CreateGif([df], 'Sym.gif')
    CreateGif([df], 'Pressure.gif', 'GetAutoGradPressure', 0, 50)

