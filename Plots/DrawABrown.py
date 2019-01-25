from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from copy import copy
import sys

from Utilities import Utilities as utl
from DrawAllEOS import ContourToPatches
from MakeSkyrmeFileBisection import LoadSkyrmeFile
from Plots.DrawLambdaRadius import DrawTightLIGO


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 8))

    xvalue = 'R(1.4)'
    yvalue = 'lambda(1.4)'
    xlower = 10.
    xupper = 14.
    ylower = 0.
    yupper = 700
    if len(sys.argv) >= 3:
        xvalue = sys.argv[1]
        yvalue = sys.argv[2]
        if len(sys.argv) == 7:
            xlower = float(sys.argv[3])
            xupper = float(sys.argv[4])
            ylower = float(sys.argv[5])
            yupper = float(sys.argv[6])
        
    # load Skyrmes
    df = LoadSkyrmeFile('Results/ABrownUDen.csv')
    df = pd.concat([df, LoadSkyrmeFile('Results/ABrownNewNoPolyTrope.csv')])

    Lambda = {}
    Radius = {}
    for key, value in df.iterrows():
        last_letter = key[-1]
        if last_letter not in Lambda:
            Lambda[last_letter] = []
            Radius[last_letter] = []
        Lambda[last_letter].append(value[yvalue])
        Radius[last_letter].append(value[xvalue])

    for last_letter in Lambda:
        if(last_letter == 'u' or last_letter == 'q'):
            if(last_letter == 'u'):
                color = 'r'
            else:
                color = 'b'
            ax.plot(Radius[last_letter], Lambda[last_letter], 'o', label='Group %s' % last_letter, color=color)

    if len(sys.argv) < 3:
        DrawTightLIGO(ax, color='aqua', alpha=0.5, fill=True)

    ax.set_xlim([xlower, xupper])
    ax.set_ylim([ylower, yupper])
    ax.set_ylabel(r'$\Lambda (1.4 M_{\odot})$')#yvalue)
    ax.set_xlabel(xvalue)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.legend()
    plt.show()
