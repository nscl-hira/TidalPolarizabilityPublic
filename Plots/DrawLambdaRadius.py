import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from copy import copy

from Utilities import Utilities as utl
from Plots.DrawAllEOS import ContourToPatches
from MakeSkyrmeFileBisection import LoadSkyrmeFile

def DrawAnnala(ax, **kwargs):
    x = np.linspace(8, 15, 100)
    y = 2.88e-6*np.power(x, 7.5)
    return ax.plot(x, y, **kwargs)

def DrawNoCrust(ax, **kwargs):
    x = np.linspace(8, 15, 100)
    y = 2.9158e-5*np.power(x, 6.6298)
    return ax.plot(x, y, **kwargs)

def DrawTightLIGO(ax, **kwargs):
    # newest constraint from LIGO group
    x = [10.5, 13.3,  13.3 ,10.5, 10.5]
    y = [70, 70, 580, 580, 70]
    path, LIGO_tight = ContourToPatches(x, y, **kwargs)
    return ax.add_patch(copy(LIGO_tight))

def DrawLooseLIGO(ax, **kwargs):
    x = [8, 15, 15, 8, 8]
    y = [70, 70, 720, 720, 70]
    path, LIGO_Loose = ContourToPatches(x, y, **kwargs)
    return ax.add_patch(copy(LIGO_Loose))


def DrawFSUGold(ax, **kwargs):
    x = [12.52, 12.915, 13.163, 13.601, 14.072, 14.122, 14.387, 14.412, 14.685, 14.818]
    y = [506., 630.16, 647.34, 737.83, 865.85, 884.65, 989.92, 978.71, 1112.1, 1288.4]
    return ax.scatter(x, y, **kwargs)


def DrawOurData(df, ax, **kwargs):
    return ax.scatter(df['R(1.4)'], df['lambda(1.4)'], **kwargs)

# ==================================
# ==================================
# Modify the legend so 2 markers can share a single label
# ==================================
# ==================================
class data_handler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        scale = fontsize / 22
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch_sq = mpatches.Circle([x0, y0 + height/2], height/2 * scale, facecolor='white', linewidth=3,
                edgecolor='black', transform=handlebox.get_transform())
        patch_circ = mpatches.Circle([x0 + width - height/2, y0 + height/2], height/2 * scale, facecolor='white',linewidth=3,
                edgecolor='magenta', transform=handlebox.get_transform())

        handlebox.add_artist(patch_sq)
        handlebox.add_artist(patch_circ)
        return patch_sq


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spines['top'].set_zorder(20)

    # load Skyrmes
    df = LoadSkyrmeFile('Results/Orig_mm2.17.csv')

    # draw from bottom layer to the top
    #DrawLooseLIGO(ax, color='aqua', alpha=0.5, fill=True)
    DrawTightLIGO(ax, color='aqua', alpha=0.5, fill=True)
    annala = DrawAnnala(ax, color='b', label='Annala et. al.', zorder=12, linewidth='3')
    no_crust = DrawNoCrust(ax, color='b', label='Skyrme without crust', zorder=13, linestyle='--', linewidth=3)
    fsu_gold = DrawFSUGold(ax, marker='s', color='r', facecolor='white', zorder=11, label='RMF based EoS', linewidth=3, s=250)
    all_data = DrawOurData(df, ax, marker='o', color='magenta', facecolor='white', zorder=10, linewidth=3, s=250, label='Skyrme based EoS')
    #agree_low_den = DrawOurData(df[df['AgreeLowDensity'] == True], ax, marker='o', color='black', facecolor='white', zorder=10, linewidth=3, s=250, label='')

    ax.set_xlim([8, 15])
    ax.set_ylim([0, 1300])
    ax.set_xlabel('Neutron Star Radius (km)')
    ax.set_ylabel(r'Tidal Deformability $\Lambda$')
    #legend = ax.legend([annala[0], no_crust[0], fsu_gold, all_data], ['Annala et. al.', 'Skyrme without crust', 'RMF based EoS', 'Skyrme based EoS'], loc='upper left', fontsize=25, handler_map={all_data: data_handler()})
    legend = ax.legend(loc='upper left', fontsize=25)
    # align marker color to text color
    for text, handle, label in zip(legend.get_texts(), *ax.get_legend_handles_labels()):
        if hasattr(handle, 'get_color'):
            color = handle.get_color()
        else:
            color = colors.rgb2hex(handle.get_edgecolor()[0]) # for scatter plot
        text.set_color(color)

    ax.set_xticks([8, 10, 12, 14])
    ax.set_yticks([0, 500, 1000])
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.show()
