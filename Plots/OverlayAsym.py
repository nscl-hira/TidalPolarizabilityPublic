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
data['Mass Skyrme'] = DataStruct("0.63 0.03 24.7 0.8", "blue", "blue", "s", "Mass(Skyrme)")
data['IAS'] = DataStruct("0.66 0.04 25.5 1.1", "blue", "white", "^", "IAS")
data['HIC isodiff'] = DataStruct("0.24 0.07 10.6 1", "magenta", "white", "*", "HIC(isodiff)")
data['HIC n/p'] = DataStruct("0.43 0.05 16.8 1.2", "magenta", "magenta", "*", "HIC(n/p)")
#data['PREXII'] = DataStruct("1 0 38.09 4.73", "red", "red", "v", "PREX-II")
data['Pion'] = DataStruct("1.45 0.1 52 13", "red", "white", "v", r"HIC($\pi$)")
#data['Pion'] = DataStruct("1.5 0.1 45.7 7.9", "red", "white", "v", r"HIC($\pi$)")
data['Mass DFT'] = DataStruct("0.72 0.01 25.4 1.1", "blue", "blue", "o", "Mass(DFT)")
data['polarizability'] = DataStruct("0.31 0.03 15.9 1", "green", "green", "D", r"$\alpha_D$")

if len(sys.argv) == 3:
    with open(sys.argv[1], 'rb') as fid:
        fig = pickle.load(fid)
    
    ax = fig.axes[0]
    pdf_name = sys.argv[2]
else:
    fig, ax = plt.subplots(figsize=(11, 9))
    pdf_name = sys.argv[1]

plots = []
labels = []
for key, content in data.items():
    value = [float(text) for text in content.data.split()]
    value[0] = value[0]*0.16
    value[1] = value[1]*0.016
    p = ax.errorbar(x=value[0], y=value[2], yerr=value[3], 
                    markerfacecolor=content.facecolor,
                    ecolor=content.edgecolor,
                    markeredgecolor=content.edgecolor,
                    marker=content.style,
                    markersize=20 if content.style != "*" else 30,
                    elinewidth=2,
                    markeredgewidth=2,
                    capsize=2,
                    capthick=2,
                    linewidth=0)
    labels.append(content.label)
    plots.append(p)

# Bill's fit
content = """ 0 0 0
0.01    2.99  3.33
0.02    5.26  5.87
0.02    7.35  8.19
0.03    9.32 10.34
0.04   11.22 12.37
0.05   13.05 14.28
0.06   14.83 16.09
0.06   16.56 17.81
0.07   18.24 19.44
0.08   19.87 21.00
0.09   21.45 22.48
0.10   22.95 23.92
0.10   24.37 25.34
0.11   25.67 26.76
0.12   26.85 28.19
0.13   27.92 29.62
0.14   28.89 31.05
0.14   29.77 32.46
0.15   30.57 33.86
0.16   31.28 35.23
0.17   31.92 36.59
0.18   32.48 37.92
0.18   32.95 39.24
0.19   33.36 40.53
0.20   33.68 41.80
0.21   33.93 43.06
0.22   34.10 44.29
0.22   34.20 45.50
0.23   34.23 46.69
0.24   34.18 47.86
0.25   34.05 49.01
0.26   33.86 50.13
0.26   33.58 51.24
0.27   33.24 52.33
0.28   32.82 53.40
0.29   32.33 54.45
0.30   31.76 55.48
0.30   31.12 56.49
0.31   30.41 57.49
0.32   29.63 58.46"""

content = np.array([[float(value) for value in line.split()] for line in content.split('\n')])
#ax.fill_between(np.linspace(0, 2, content.shape[0])*0.16, content[:, 1], content[:, 2], facecolor='none', edgecolor='red', alpha=1, label="Quadratic best fit")
ax.set_xlabel(r'$\rho$ (fm$^{-3}$)')
ax.set_ylabel(r'S($\rho$) (MeV)')
plt.xlim(0, 2.5*0.16)
plt.ylim(0, 80)
#plt.yscale('log')
plt.ylim(bottom=1)

leg = plt.legend(loc='upper left')
for h, t in zip(leg.legendHandles, leg.texts):
    plots = [h] + plots
    labels = [t.get_text()] + labels


halfPlots = int(len(plots)/2)
legend1 = plt.legend([plots[i] for i in range(halfPlots)], [labels[i] for i in range(halfPlots)], loc='upper left', frameon=False, fontsize=20)
plt.legend([plots[i] for i in range(halfPlots, len(plots))], [labels[i] for i in range(halfPlots, len(plots))], loc='lower right', frameon=False, fontsize=20)
#plt.legend(frameon=False, fontsize=20, ncol=2, loc='upper left')
plt.gca().add_artist(legend1)
fig = mpl.pyplot.gcf()
fig.set_size_inches(11, 9)
plt.subplots_adjust(right=0.95)
plt.savefig(pdf_name)
