import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import pickle
from collections import namedtuple
from Plots.FillableHist import FillableHist2D
import numpy as np
from Utilities.Utilities import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['errorbar.capsize'] =  2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

DataStruct = namedtuple("DataStruct", ["data", "edgecolor", "facecolor", "style", "label"])

data = {}
data['RileyM14'] = DataStruct("12.71 1.14 1.19 1.34 0.15 0.16", "red", "white", "*", r"Riley $1.4M_\odot$")

data['RileyM2'] = DataStruct("12.39 1.3 0.98 2.07 0.07 0.07", "red", "white", "s", r"Riley $2.1M_\odot$")
data['MillerM14'] = DataStruct("13.02 1.24 1.06 1.44 0.15 0.14", "black", "white", "*", r"Miller $1.4M_\odot$")
data['MillerM2'] = DataStruct("13.7 2.6 1.5 2.08 0.07 0.07", "black", "white", "s", r"Miller $2.1M_\odot$")
#new data here


if len(sys.argv) == 3:
    with open(sys.argv[1], 'rb') as fid:
        fig = pickle.load(fid)
    
    ax = fig.axes[0]
    pdf_name = sys.argv[2]
else:
    fig, ax = plt.subplots(figsize=(11, 9))
    pdf_name = sys.argv[1]

for key, content in data.items():
    value = [float(text) for text in content.data.split()]
    p = ax.errorbar(x=value[0], xerr=[[value[2]], [value[1]]], y=value[3], yerr=[[value[5]], [value[4]]], 
                    markerfacecolor=content.facecolor,
                    ecolor=content.edgecolor,
                    markeredgecolor=content.edgecolor,
                    marker=content.style,
                    markersize=20 if content.style != "*" else 30,
                    elinewidth=2,
                    markeredgewidth=2,
                    capsize=2,
                    capthick=2,
                    label=content.label,
                    linewidth=0)

plt.legend(frameon=False, fontsize=20, loc='upper right')
fig = mpl.pyplot.gcf()
fig.set_size_inches(11, 9)
plt.subplots_adjust(right=0.95)
plt.savefig(pdf_name)
