import numpy as np
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
import matplotlib.pyplot as plt

import SkyrmeEOS as sky
from Constants import rho0

def GetContour(df, rho_min, rho_max):
    n = np.linspace(rho_min,rho_max,1000)
    value = []
    for index, row in df.iterrows():
        value.append(sky.GetEnergy(n*rho0, 0., row))
    value = np.array(value)
    print(value.shape)
    contour = np.concatenate([np.amax(value, axis=0), np.amin(value, axis=0)[::-1]]).flatten()
    values = np.array([n, n[::-1]]).flatten()
    # complete the loop by including the first element as the last
    contour = np.append(contour, contour[0])
    values = np.append(values, values[0])
    return values, contour
    

def PlotSkyrmeSymEnergy(df, ax, color=''):
    n = np.linspace(0,3,1000)
    for index, row in df.iterrows():
        ax.plot(n, sky.GetAsymEnergy(n*rho0, row), label=index, zorder=-32, color=color)
    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_ylabel('$S(\\rho)$ (MeV)', fontsize=30)
    return ax

def PlotSkyrmeEnergy(df, ax, color=''):
    # ax = plt.subplot(111)
    n = np.linspace(0, 3, 1000)
    #for index, row in df.iterrows():
    #    energy = sky.GetEnergy(n*rho0, 0.5, row)
    #    ax.plot(n, energy, label='%s SNM' % index, color=color)
    for index, row in df.iterrows():
        energy = sky.GetEnergy(n*rho0, 0., row)
        ax.plot(n, energy, label='%s PNM' % index, color=color)
    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_ylabel('Energy per nucleons (MeV)', fontsize=30)
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    #plt.subplots_adjust(right=0.85)
    return ax

def PlotSkyrmePressure(df, ax, color=''):
    n = np.linspace(0, 3, 1000)
    for index, row in df.iterrows():
        pressure = sky.GetAutoGradPressure(n*rho0, 0.0, row)
        ax.plot(n, pressure, label=index, color=color)
    ax.set_ylim([1,1000])
    ax.set_xlim([0,3])
    ax.set_yscale('log')
    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_ylabel('Pressure (MeV/fm3)', fontsize=30)
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    return ax

def PlotMassVsRadius(mass, radius, ax, color=''):
    # ax = plt.subplot(111)
    for key in mass:
        ax.plot(radius[key], mass[key], label=key, color=color)
    ax.set_xlabel('Radius (km)', fontsize=30)
    ax.set_ylabel('Mass (solar)', fontsize=30)
    ax.set_ylim([0, 3])
    ax.set_xlim([0, 20])
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    return ax

def PlotMassVsLambda(mass, lambda_, ax, color=''):
    # ax = plt.subplot(111)
    for key in mass:
        ax.plot(mass[key], lambda_[key], label=key, color=color)
    ax.set_xlabel('Mass (solar)', fontsize=30)
    ax.set_ylabel('lambda', fontsize=30)
    #ax.set_yscale('log')
    ax.set_ylim([0, 4000])
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    #plt.subplots_adjust(right=0.85)
    #plt.show()
    return ax

def PlotLambdaRadius(mass, radius, lambda_, ax, color=''):
    # ax = plt.subplot(111)
    for key in mass:
        # trying to find the radius and the polarizability at 1.4 solar mass
        # using simple linear interpolation
        ax.plot([np.interp(1.4, mass[key], radius[key])], [np.interp(1.4, mass[key], lambda_[key])], label=key, marker=marker.next(), markersize=20, color=color)
    ax.set_xlabel('R(1.4 solar mass) km', fontsize=30)
    ax.set_ylabel('lambda', fontsize=30)
    ax.set_ylim([-200, 2000])
    ax.set_xlim([0, 20])
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    #plt.subplots_adjust(right=0.85)
    #plt.show()
    return ax
    
