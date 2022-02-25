import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['figure.figsize'] = (9.7082039325, 6.0)
matplotlib.rcParams['xtick.labelsize'] = 27.0
matplotlib.rcParams['ytick.labelsize'] = 27.0
matplotlib.rcParams['axes.labelsize'] = 27.0
matplotlib.rcParams['legend.fontsize'] = 22.0
matplotlib.rcParams['font.family']= 'Times New Roman'
matplotlib.rcParams['font.sans-serif']= ['Bitstream Vera Sans']
#matplotlib.rcParams['text.usetex']= True
matplotlib.rcParams['mathtext.fontset']= 'stixsans'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True

import lal
import scipy
import random
from random import randint
import csv
from scipy.interpolate import interp1d

c_cgs=lal.C_SI*100
rhonuc=2.8e14


fig = plt.figure(figsize=(9.7082039325, 6))

# Make the example pressure-baryon density plot
def example_p_rho_plot():
    pathprefix='Quantiles/p_rho/'

    # extract the data
    path=pathprefix+"prior_prho_quantiles.csv"
    columns = open(path, 'r').readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    baryon_density = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    # We rely on the fact that the 50th row contains the 50th percentile, etc.
    pressures50 = data[50,:]*c_cgs*c_cgs
    pressures5 = data[5,:]*c_cgs*c_cgs
    pressures95 = data[95,:]*c_cgs*c_cgs

    # create the bounding 5 % and 95 % curves, and fill the interior
    plt.fill_between(baryon_density,pressures5[1:],pressures95[1:],color='c',alpha=0.05)
    plt.plot(baryon_density,pressures5[1:],c='k',lw=2.5,label='Prior')
    plt.plot(baryon_density,pressures95[1:],c='k',lw=2.5)

    # Repeat
    path=pathprefix+"no_j0740_prho_quantiles.csv"
    columns = open(path, 'r').readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    baryon_density = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    pressures50 = data[50,:]*c_cgs*c_cgs
    pressures5 = data[5,:]*c_cgs*c_cgs
    pressures95 = data[95,:]*c_cgs*c_cgs
    
    plt.fill_between(baryon_density,pressures5[1:],pressures95[1:],color='c',alpha=0.2)
    plt.plot(baryon_density,pressures5[1:],c='c',lw=2.5,label='PSR + GW + J0030')
    plt.plot(baryon_density,pressures95[1:],c='c',lw=2.5)
    
    path=pathprefix+"all_miller_prho_quantiles.csv"
    columns = open(path, 'r').readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    baryon_density = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    pressures50 = data[50,:]*c_cgs*c_cgs
    pressures5 = data[5,:]*c_cgs*c_cgs
    pressures95 = data[95,:]*c_cgs*c_cgs

    plt.fill_between(baryon_density,pressures5[1:],pressures95[1:],alpha=0.5)
    plt.plot(baryon_density,pressures5[1:],c='b',lw=2.5,label='PSR + GW + J0030 + J0740')
    plt.plot(baryon_density,pressures95[1:],c='b',lw=2.5)

    # Finishing up
    plt.axvline(rhonuc,c='k')
    plt.axvline(2*rhonuc,c='k')
    plt.axvline(6*rhonuc,c='k')

    plt.tick_params(direction='in')
    plt.yscale('log')
    plt.xscale('log')

    # Cosmetics, display the nuclear saturation density and several of its multiples
    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.ylabel('$ P \,(\mathrm{dyn/cm}^2)$')
    plt.xlabel('$\\rho \,(\mathrm{g/cm}^3)$')
    plt.xlim(4e13,2.8e15)
    plt.ylim(4e31,1e37)
    plt.legend(frameon=True,fancybox=True,framealpha=1,fontsize=18)
    plt.text(3e14,0.6e32,'$\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
    plt.text(6e14,0.8e32,'$2\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
    plt.text(18e14,0.8e32,'$6\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)

    plt.show()
    fig.savefig("pressure_density_comparison.pdf",bbox_inches='tight')
# Make the example mass-radius plot 
def example_m_r_plot():
    Msol_in_km=lal.MSUN_SI*lal.G_SI/lal.C_SI/lal.C_SI/1000

    fig = plt.figure(figsize=(9.7082039325, 6))


    pathprefix='Quantiles/m_r/'

    path=pathprefix+"prior_mr_quantiles.csv"
    columns = open(path, 'r').readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    mass = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    radius50 = data[50,:]
    radius5 = data[5,:]
    radius95 = data[95,:]

    plt.fill_betweenx(mass,radius5[1:],radius95[1:],color='k',alpha=0.05)
    plt.plot(radius5[1:],mass,c='k',lw=2.5,label='Prior')
    plt.plot(radius95[1:],mass,c='k',lw=2.5)

    path=pathprefix+"no_j0740_mr_quantiles.csv"
    columns = open(path, 'r').readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    mass = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    radius50 = data[50,:]
    radius5 = data[5,:]
    radius95 = data[95,:]

    plt.fill_betweenx(mass,radius5[1:],radius95[1:],color='darkturquoise',alpha=0.1)
    plt.plot(radius5[1:],mass,c='darkturquoise',lw=3,label='PSR + GW + J0030')
    plt.plot(radius95[1:],mass,c='darkturquoise',lw=3)



    path=pathprefix+"all_miller_mr_quantiles.csv"
    columns = open(path, 'r').readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    mass = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    radius50 = data[50,:]
    radius5 = data[5,:]
    radius95 = data[95,:]

    plt.fill_betweenx(mass,radius5[1:],radius95[1:],color='b',alpha=0.6)
    plt.plot(radius5[1:],mass,c='b',lw=2.5,label='PSR + GW + J0030 + J0740')
    plt.plot(radius95[1:],mass,c='b',lw=2.5)

    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.xlabel('$ R$ (km)')
    plt.ylabel('$M \, (M_{\odot})$')
    plt.ylim(0.75,2.25)
    plt.xlim(6,18)

    plt.fill_between(np.arange(6,20,2),2.15,2.01,color='grey',alpha=0.5)
    plt.fill_between(np.arange(6,20,2),2.08,1.97,color='grey',alpha=0.3)

    plt.text(6.5,1.99,'J0348+0432',color='k',fontsize=25)
    plt.text(6.5,2.07,'J0740+6620',color='k',fontsize=25)

    plt.yticks([1,1.4,1.8,2.2])
    plt.tick_params(direction='in')
    plt.legend(frameon=True,fancybox=True,framealpha=0.5,loc="lower right",fontsize=18)
    plt.show()
    fig.savefig("mass_radius_comparison.pdf",bbox_inches='tight')

# make the example speed of sound squared, baryone density plot
def example_cs2_rho_plot():
    fig = plt.figure(figsize=(9.7082039325, 6))
    pathprefix='Quantiles/cs2_rho/'
    path=pathprefix+"prior_cs2rho_quantiles.csv"
    columns = open(path, 'r').readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    baryon_density = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    cs2s50 = data[50,:]
    cs2s5 = data[5,:]
    cs2s95 = data[95,:]

    plt.fill_between(baryon_density,cs2s5[1:],cs2s95[1:],color='c',alpha=0.05)
    plt.plot(baryon_density,cs2s5[1:],c='k',lw=2.5,label='Prior')
    plt.plot(baryon_density,cs2s95[1:],c='k',lw=2.5)

    path=pathprefix+"no_j0740_cs2rho_quantiles.csv"
    columns = open(path, 'r').readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    baryon_density = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    cs2s50 = data[50,:]
    cs2s5 = data[5,:]
    cs2s95 = data[95,:]

    plt.fill_between(baryon_density,cs2s5[1:],cs2s95[1:],color='c',alpha=0.2)
    plt.plot(baryon_density,cs2s5[1:],c='c',lw=2.5,label='PSR + GW + J0030')
    plt.plot(baryon_density,cs2s95[1:],c='c',lw=2.5)

    path=pathprefix+"all_miller_cs2rho_quantiles.csv"
    columns = open(path, 'r').readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    baryon_density = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    cs2s50 = data[50,:]
    cs2s5 = data[5,:]
    cs2s95 = data[95,:]

    plt.fill_between(baryon_density,cs2s5[1:],cs2s95[1:],alpha=0.5)
    plt.plot(baryon_density,cs2s5[1:],c='b',lw=2.5,label='PSR + GW + J0030 + J0740')
    plt.plot(baryon_density,cs2s95[1:],c='b',lw=2.5)

    plt.axvline(rhonuc,c='k')
    plt.axvline(2*rhonuc,c='k')
    plt.axvline(6*rhonuc,c='k')
    plt.axhline(1/3, c='k')

    plt.tick_params(direction='in')

    plt.xscale('log')

    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.ylabel('$ c_s^2/c^2$')
    plt.xlabel('$\\rho \,(\mathrm{g/cm}^3)$')
    plt.xlim(4e13,2.8e15)
    plt.ylim(0.0,1.0)
    plt.legend(frameon=True,fancybox=True,framealpha=1,fontsize=18)
    plt.text(2.8e14,0.86,'$\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
    plt.text(4.65e14,0.86,'$2\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
    plt.text(14e14,0.86,'$6\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
    plt.text(5e13,0.37,'$c_s^2/c^2 = 1/3$',fontsize=22)

    plt.show()
    fig.savefig("cs2_rho_comparison.pdf",bbox_inches='tight')

if __name__ == "__main__":
    # Uncomment lines to run
    example_p_rho_plot()
    #example_cs2_rho_plot()
    #example_m_r_plot()
