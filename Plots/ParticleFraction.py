from copy import copy
import numpy as np
import matplotlib as mpl
import pandas as pd
#mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
from tqdm import tqdm

from Utilities.EOSDrawer import EOSDrawer
from MakeSkyrmeFileBisection import LoadSkyrmeFile, CalculatePolarizability
from Utilities.Constants import *
from Utilities.SkyrmeEOS import Skryme


if __name__ == '__main__':
    df = LoadSkyrmeFile('Results/Orig_mm2.17.csv')
    df = df[df['NegSound']==False]

    fig, ax = plt.subplots()

    drawer = EOSDrawer(df)

    for index, row in tqdm(df.iterrows(), total=df.shape[0], ncols=100):
        rho, pfrac, mufrac = drawer.ParticleFraction(index)
        efrac = pfrac - mufrac
        rho = rho/0.16
        mask = (rho > 1) & (rho < 3)

        labels = ['Electron fraction', 'Muon fraction']
        ax.stackplot(rho[mask], efrac[mask], mufrac[mask], labels=labels)
        ax.set_xlim([1., 3.])
        ax.set_xlabel(r'$\rho/\rho_{0}$')
        ax.set_ylabel('Particle fraction')
        leg = ax.legend(shadow=True, loc='lower left', facecolor='w', framealpha=1)
        
        plt.savefig('Plots/PFrac/%s.png' % index)
        ax.clear()
