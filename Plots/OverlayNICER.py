import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import namedtuple

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['errorbar.capsize'] =  2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

c_cgs = 29979245800.0
to_mev = 0.1/1.60218e32
to_fm3 = 0.16/2.8e14

#with open('PressurePost.pkl', 'rb') as fid:
if len(sys.argv) == 3:
    with open(sys.argv[1], 'rb') as fid:
        fig = pickle.load(fid)
    
    ax = fig.axes
    pdf_name = sys.argv[2]
else:
    fig, ax = plt.subplots(figsize=(11, 9))
    ax = [ax]
    pdf_name = sys.argv[1]

pathprefix='Plots/Quantiles/Quantiles/p_rho/'
path=pathprefix+"all_miller_prho_quantiles.csv"
columns = open(path, 'r').readline().strip().split(',')
data = np.loadtxt(path, delimiter=',', skiprows=1)
baryon_density = [float(_.split('=')[1][:-1])*to_fm3 for _ in columns[1:]]
pressures50 = data[50,:]*c_cgs*c_cgs*to_mev
pressures5 = data[5,:]*c_cgs*c_cgs*to_mev
pressures95 = data[95,:]*c_cgs*c_cgs*to_mev

plt.fill_between(baryon_density,pressures5[1:],pressures95[1:],alpha=0.5, zorder=-1, color='red', linewidth=2.5, label='PSR + GW + J0030 + J0740')
#plt.plot(baryon_density,pressures5[1:],c='b',lw=2.5,label='PSR + GW + J0030 + J0740', zorder=-1, color='red')
#plt.plot(baryon_density,pressures95[1:],c='b',lw=2.5, zorder=-1, color='red')

ax[0].set_xlabel(r'$\rho$ (fm$^{-3}$)')
ax[0].set_ylabel(r'P$_{NS}$($\rho$) (MeV/fm$^3$)')

plt.legend(frameon=False, fontsize=20, loc='upper left')
ax[0].set_xlim(1e-2,2*0.16)
ax[0].set_ylim(1e-1, 1e2)
fig = mpl.pyplot.gcf()
fig.set_size_inches(11, 9)

plt.savefig(pdf_name)
