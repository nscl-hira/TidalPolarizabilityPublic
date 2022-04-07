import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import namedtuple

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['errorbar.capsize'] =  2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

#with open('PressurePost.pkl', 'rb') as fid:
with open(sys.argv[1], 'rb') as fid:
    fig = pickle.load(fid)

ax = fig.axes

pathprefix='Plots/Quantiles/Quantiles/m_r/'
path=pathprefix+"all_miller_mr_quantiles.csv"
columns = open(path, 'r').readline().strip().split(',')
data = np.loadtxt(path, delimiter=',', skiprows=1)
mass = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
radius50 = data[50,:]
radius5 = data[5,:]
radius95 = data[95,:]

plt.fill_betweenx(mass,radius5[1:],radius95[1:],color='red', linewidth=2.5, alpha=0.5, zorder=-1, label='PSR + GW + J0030 + J0740')
ax[0].set_xlabel(r'$R$ (km)')
ax[0].set_xlim(10, 20)
ax[0].set_ylim(1, 2)
ax[0].set_ylabel(r'$M (M_\odot$)')

plt.legend(frameon=False, fontsize=20, loc='upper right')
fig = mpl.pyplot.gcf()
fig.set_size_inches(11, 9)

plt.savefig(sys.argv[2])
