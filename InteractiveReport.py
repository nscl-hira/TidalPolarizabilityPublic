import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MakeSkyrmeFileBisection import LoadSkyrmeFile
from Utilities.EOSDrawer import EOSDrawer 

def DrawEOS(row, ax, xlim=[1e-2, 1e4], ylim=[1e-4, 1e4], **kwargs):
    drawer.DrawEOS(df=row, ax=ax, xlim=xlim, ylim=ylim, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Energy\ Density\ (MeV\ fm^{-3})$')
    ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
    if 'xname' in kwargs:
        ax.set_xlabel(kwargs['xname'])
    if 'yname' in kwargs:
        ax.set_ylabel(kwargs['yname'])


def update_annot(ind):
    new_ax[0].clear()
    new_ax[1].clear()
    new_ax[2].clear()
    DrawEOS(df.ix[[df.index.values[ind['ind'][0] + 1]]], new_ax[0])
    DrawEOS(df.ix[[df.index.values[ind['ind'][0] + 1]]], new_ax[1], xname='rho/rho0', yname='GetPressure', xlim=[1e-8, 6], ylim=[1e-4, 1e4])
    DrawEOS(df.ix[[df.index.values[ind['ind'][0] + 1]]], new_ax[2], xname='rho', yname='GetEnergyDensity', xlim=[1e-8, 10*0.16], ylim=[1e-2, 1e4])

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([df.index.values[n+1] for n in ind["ind"]]))
    hover_x = [df['R(1.4)'].iloc[n+1] for n in ind["ind"]]
    hover_y = [df['lambda(1.4)'].iloc[n+1] for n in ind["ind"]]
    highlight.set_offsets(np.c_[hover_x, hover_y])
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            highlight.set_visible(True)
            for n_fig in new_fig:
                n_fig.canvas.draw_idle()
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                highlight.set_visible(False)
                for n_fig in new_fig:
                    n_fig.canvas.draw_idle()
                fig.canvas.draw_idle()

if __name__ == '__main__':
    df = LoadSkyrmeFile('Results/test.csv')
    drawer = EOSDrawer(df)
    
    fig, ax = plt.subplots()
    sc = ax.scatter(df['R(1.4)'], df['lambda(1.4)'], marker='o')
    ax.set_xlim([9, 16])
    ax.set_ylim([0, 1500])
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    hover_x, hover_y = [], []
    highlight = ax.scatter(hover_x, hover_y, facecolor=None, edgecolor='r')
    highlight.set_visible(False)
    
    new_fig, new_ax = [], []
    for i in range(0, 3):
        f, a = plt.subplots()
        new_fig.append(f)
        new_ax.append(a)
    
    DrawEOS(df.ix[[df.index.values[0]]], new_ax[0])
    DrawEOS(df.ix[[df.index.values[0]]], new_ax[1], xname='rho/rho0', yname='GetPressure', xlim=[1e-8, 6], ylim=[1e-4, 1e4])
    DrawEOS(df.ix[[df.index.values[0]]], new_ax[2], xname='rho', yname='GetEnergyDensity', xlim=[1e-8, 10*0.16], ylim=[1e-2, 1e4])
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()
