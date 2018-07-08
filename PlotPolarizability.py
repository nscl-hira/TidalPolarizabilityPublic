from ROOT import TGraph, TCanvas
from ROOT import gROOT, gPad, gStyle
from array import array

from scipy.optimize import curve_fit
import itertools
color = itertools.cycle(('g', 'purple', 'r', 'black', 'grey', 'orange', 'b')) 
linestyle = itertools.cycle(('dashdot','dashed','dotted')) 
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import matplotlib.patches as patches
import autograd.numpy as np
import pandas as pd
from copy import deepcopy, copy

import Utilities.Utilities as utl
from Utilities.Constants import *

def power_law(x, a, b, c):
    return a*np.power(x, b) + c
   # obs -- observed value
    # exp -- expected value

def PlotPolarizability(df, ax, label):
    ax.plot(df['R(1.4)'], df['lambda(1.4)'], 'ro', label=label, color=color.next(), markerfacecolor='w')

    df = df[(df['R(1.4)'] < 16) & (df['R(1.4)'] > 8)]
    #popt, pcov = curve_fit(power_law, df['R(1.4)'], df['P(2rho0)'])
    #xaxis = np.linspace(0, 16, 100)
    #ax.plot(xaxis, power_law(xaxis, *popt), label=r'$fit: {:.2e}R^{{{:4.2f}}} + {:4.2f}$'.format(*popt))

    ax.set_xlabel(r'Neutron Star Radius (km)')
    ax.set_ylabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
    ax.set_xlim([8, 16])
    #ax.set_ylim([0, 1600])
    

if __name__ == "__main__":
    ax = plt.subplot(111)

    file_list = {#'Just Skyrme':'Results/Skyrme_Only.csv',
                 #'r$td=0.2, sd=0.7$': 'Results/Skyrme_td_0.2_sd_0.7.csv',
                 #'r$td=0.2, sd=0.5$': 'Results/Skyrme_td_0.2_sd_0.5.csv',
                 #'r$td=0.2, sd=0.3$': 'Results/Skyrme_td_0.2_sd_0.3.csv',
                 #'r$td=0.1, sd=0.7$': 'Results/Skyrme_td_0.1_sd_0.7.csv',
                 #'r$td=0.1, sd=0.5$': 'Results/Skyrme_td_0.1_sd_0.5.csv',
                 #'r$td=0.1, sd=0.3$': 'Results/Skyrme_td_0.1_sd_0.3.csv',
                 #'PRC 60%': 'Results/Skyrme_pd_0.6.csv',
                 #'PRC 70%': 'Results/Skyrme_pd_0.7.csv',
                 #'PRC 80%': 'Results/Skyrme_pd_0.8.csv',
                 #'PRC 90%': 'Results/Skyrme_pd_0.9.csv',
                 'New': 'test.csv'
                 }
    for key, file_ in file_list.iteritems():           
        df = pd.read_csv(file_, index_col=0)
        df.fillna(0, inplace=True)
        PlotPolarizability(df.loc[df['ViolateCausality']==True], ax, 'Acausal')
        PlotPolarizability(df.loc[df['ViolateCausality']==False], ax, 'Causal')
     
    ax.legend()
    
    plt.show()

    num_files = len(file_list)
    #fig, ax = plt.subplots(num_files, num_files)
    c1 = TCanvas('c1', 'Effect of transition density', 200, 10, 700, 500)
    c1.Divide(num_files, num_files)
    graph = []
    slopes = []
    gStyle.SetOptFit(1111)

    row = 0
    for keyrow, filerow in file_list.iteritems():
        keyrow = keyrow + " lambda(1.4)"
        col = 0
        col_slopes = {}
        for keycol, filecol in copy(file_list).iteritems():
            keycol = keycol + " lambda(1.4)"
            c1.cd(col + num_files*row + 1)
            #local_ax = ax[row][col]
            #local_ax.xaxis.set_label_position('bottom')

            df_row = pd.read_csv(filerow, index_col=0)
            df_row.fillna(0, inplace=True)

            df_col = pd.read_csv(filecol, index_col=0)
            df_col.fillna(0, inplace=True)

            common_skyrmes = list(set(df_row.index).intersection(set(df_col.index)))
            row_data = df_row.loc[common_skyrmes, 'lambda(1.4)'].tolist()
            col_data = df_col.loc[common_skyrmes, 'lambda(1.4)'].tolist()
           
            graph.append(TGraph(len(row_data), array('d', row_data), array('d', col_data)))
            graph[-1].Draw("AP")
            graph[-1].Fit("pol1")
            graph[-1].SetMarkerStyle(8)
            graph[-1].SetTitle('')
            graph[-1].GetXaxis().SetTitle(keyrow)
            graph[-1].GetYaxis().SetTitle(keycol)
            tf1 = gROOT.FindObject("pol1")
            col_slopes[keycol] = tf1.GetParameter(1)
            col_slopes['row name'] = keyrow
            """
            local_ax.tick_params('both', length=0, width=0, which='major')
            local_ax.tick_params('both', length=0, width=0, which='minor')
            
            local_ax.plot(df_row.loc[common_skyrmes, 'R(1.4)'], df_col.loc[common_skyrmes, 'R(1.4)'], 'ro')
            if row == num_files - 1:
                local_ax.set_xlabel(keycol, fontsize=15)
            else:
                local_ax.set_xticklabels([])
            if col == 0:
                local_ax.set_ylabel(keyrow, fontsize=15)
            else:
                local_ax.set_yticklabels([])
            if col == 0 and row == 0:
                local_ax.set_yticklabels(local_ax.get_xticks().tolist())
            """
            col = col + 1
        slopes.append(col_slopes)
        row = row + 1
           
    slopes = pd.DataFrame(slopes).set_index('row name')
    print(slopes)
    c1.Update()

    gPad.WaitPrimitive()
    #plt.subplots_adjust(wspace=0, hspace=0)
    #plt.show()

