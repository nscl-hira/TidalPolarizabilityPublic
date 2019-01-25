from copy import copy
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import pandas as pd

from Utilities.EOSDrawer import EOSDrawer
import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *

if __name__ == "__main__":
    df = pd.read_csv('Results/Newest.csv', index_col=0)
    df.fillna(0, inplace=True)

    ax1, ax2 = utl.PlotMaster(df, [], [], (), pfrac=0.5)
    plt.show()

    # Plot all the EOSs
    drawer = EOSDrawer(df)
    ax = plt.subplot(111)
    drawer.DrawEOS(ax=ax, xlim=[1e-2, 1e4], ylim=[1e-4, 1e4])
    ax.set_xlabel(r'$Energy\ Density\ (MeV\ fm^{-3})$')
    ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
    plt.show()

