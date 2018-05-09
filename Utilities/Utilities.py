import numpy as np
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
import matplotlib.pyplot as plt

import SkyrmeEOS as sky
from Constants import rho0

def PlotSkyrmeSymEnergy(fig, df):
    ax = plt.subplot(111)
    n = np.linspace(0,3,1000)
    for index, row in df.iterrows():
        ax.plot(n, sky.GetAsymEnergy(n*rho0, row), label=index, zorder=-32, color='b')
    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_ylabel('$S(\\rho)$ (MeV)', fontsize=30)
    plt.subplots_adjust(right=0.85)
    return ax

def PlotSkyrmeEnergy(df):
    fig = plt.figure()
    ax = plt.subplot(111)
    n = np.linspace(0, 5, 100)
    for index, row in df.iterrows():
        energy = sky.GetEnergy(n*rho0, 0.5, row)
        ax.plot(n, energy, label='%s SNM' % index, marker=marker.next())
    for index, row in df.iterrows():
        energy = sky.GetEnergy(n*rho0, 0., row)
        ax.plot(n, energy, label='%s PNM' % index, marker=marker.next())
    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_ylabel('Energy per nucleons (MeV)', fontsize=30)
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    plt.subplots_adjust(right=0.85)
    plt.show()

def PlotSkyrmePressure(df):
    ax = plt.subplot(111)
    n = np.linspace(0, 5, 100)
    for index, row in df.iterrows():
        pressure = sky.GetAutoGradPressure(n*rho0, 0.0, row)
        ax.plot(n, pressure, label=index, marker=marker.next())
    ax.set_ylim([1,1000])
    ax.set_xlim([1,5])
    ax.set_yscale('log')
    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_ylabel('Pressure (MeV/fm3)', fontsize=30)
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    plt.subplots_adjust(right=0.85)
    plt.show()

def PlotMassVsRadius(mass, radius):
    ax = plt.subplot(111)
    for key in mass:
        ax.plot(radius[key], mass[key], label=key, marker=marker.next())
    ax.set_xlabel('Radius (km)', fontsize=30)
    ax.set_ylabel('Mass (solar)', fontsize=30)
    ax.set_ylim([0, 3])
    ax.set_xlim([0, 20])
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    plt.subplots_adjust(right=0.85)
    plt.show()

def PlotMassVsLambda(mass, lambda_):
    ax = plt.subplot(111)
    for key in mass:
        ax.plot(mass[key], lambda_[key], label=key, marker=marker.next())
    ax.set_xlabel('Mass (solar)', fontsize=30)
    ax.set_ylabel('lambda', fontsize=30)
    #ax.set_yscale('log')
    ax.set_ylim([0, 4000])
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    plt.subplots_adjust(right=0.85)
    plt.show()

def PlotLambdaRadius(mass, radius, lambda_):
    ax = plt.subplot(111)
    for key in mass:
        # trying to find the radius and the polarizability at 1.4 solar mass
        # using simple linear interpolation
        ax.plot([np.interp(1.4, mass[key], radius[key])], [np.interp(1.4, mass[key], lambda_[key])], label=key, marker=marker.next(), markersize=20)
    ax.set_xlabel('R(1.4 solar mass) km', fontsize=30)
    ax.set_ylabel('lambda', fontsize=30)
    ax.set_ylim([-200, 2000])
    ax.set_xlim([0, 20])
    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
    plt.subplots_adjust(right=0.85)
    plt.show()
    
