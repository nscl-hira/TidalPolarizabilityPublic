import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import copy

from Utilities import Utilities as utl
from DrawAllEOS import ContourToPatches
from MakeSkyrmeFileBisection import LoadSkyrmeFile

def DrawAnnala(ax, **kwargs):
    x = np.linspace(8, 15, 100)
    y = 2.88e-6*np.power(x, 7.5)
    ax.plot(x, y, **kwargs)

def DrawTightLIGO(ax, **kwargs):
    # newest constraint from LIGO group
    x = [10.5, 13.3,  13.3 ,10.5, 10.5]
    y = [70, 70, 580, 580, 70]
    path, LIGO_tight = ContourToPatches(x, y, **kwargs)
    ax.add_patch(copy(LIGO_tight))

def DrawLooseLIGO(ax, **kwargs):
    x = [8, 15, 15, 8, 8]
    y = [70, 70, 720, 720, 70]
    path, LIGO_Loose = ContourToPatches(x, y, **kwargs)
    ax.add_patch(copy(LIGO_Loose))


def DrawFSUGold(ax, **kwargs):
    x = [12.52, 12.915, 13.163, 13.601, 14.072, 14.122, 14.387, 14.412, 14.685, 14.818]
    y = [506., 630.16, 647.34, 737.83, 865.85, 884.65, 989.92, 978.71, 1112.1, 1288.4]
    ax.scatter(x, y, **kwargs)


def DrawOurData(df, ax, **kwargs):
    ax.scatter(df['R(1.4)'], df['lambda(1.4)'], **kwargs)


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spines['top'].set_zorder(20)

    # load Skyrmes
    df = LoadSkyrmeFile('Results/Orig_mm2.17.csv')

    # draw from bottom layer to the top
    DrawLooseLIGO(ax, color='aqua', alpha=0.5, fill=True)
    DrawTightLIGO(ax, color='blue', fill=False, linewidth='3')
    DrawAnnala(ax, color='r', label='Annala et. al.', zorder=12)
    DrawFSUGold(ax, marker='s', color='r', facecolor='white', zorder=11, label='RMF based EoS', linewidth='3', s=200)
    DrawOurData(df, ax, marker='o', color='magenta', facecolor='white', zorder=10, label='Skyrme based EoS', linewidth='3', s=250)

    ax.set_xlim([8, 15])
    ax.set_ylim([0, 1300])
    ax.set_xlabel('Neutron Star Radius (km)')
    ax.set_ylabel(r'Tidal Deformability $\Lambda$')
    legend = ax.legend(loc='upper left', fontsize=25)
    # align marker color to text color
    for text, color in zip(legend.get_texts(), ['r', 'r', 'magenta']):
        text.set_color(color)

    ax.set_xticks([8, 10, 12, 14])
    ax.set_yticks([0, 500, 1000])
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.show()
