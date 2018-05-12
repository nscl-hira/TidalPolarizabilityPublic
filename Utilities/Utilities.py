import numpy as np
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
import matplotlib.pyplot as plt

import SkyrmeEOS as sky
from Constants import *

def GetContour(df, rho_min, rho_max, func=sky.GetEnergy):
    n = np.linspace(rho_min,rho_max,1000)
    value = []
    for index, row in df.iterrows():
        value.append(func(n*rho0, 0., row))
    value = np.array(value)
    contour = np.concatenate([np.amax(value, axis=0), np.amin(value, axis=0)[::-1]]).flatten()
    values = np.array([n, n[::-1]]).flatten()
    # complete the loop by including the first element as the last
    contour = np.append(contour, contour[0])
    values = np.append(values, values[0])
    return values, contour
    

def PlotSkyrmeSymEnergy(df, ax, range_=[0,3], pfrac=0, **args):
    n = np.linspace(*range_, num=1000)
    for index, row in df.iterrows():
        ax.plot(n, sky.GetAsymEnergy(n*rho0, row), label=index, zorder=-32, **args)
    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_ylabel('$S(\\rho)$ (MeV)', fontsize=30)
    return ax

def PlotSkyrmeEnergy(df, ax, range_=[0,3], pfrac=0, **args):
    n = np.linspace(*range_, num=1000)
    for index, row in df.iterrows():
        energy = sky.GetEnergy(n*rho0, pfrac, row)
        labels = ax.plot(n, energy, label='%s PNM' % index, **args)
    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_ylabel('Energy per nucleons (MeV)', fontsize=30)
    return labels

def PlotSkyrmePressure(df, ax, range_=[0,3], pfrac=0, **args):
    n = np.linspace(*range_, num=1000)
    for index, row in df.iterrows():
        pressure = sky.GetAutoGradPressure(n*rho0, pfrac, row)
        ax.plot(n, pressure, label=index, zorder=1, **args)
    ax.set_ylim([1,1000])
    ax.set_xlim(range_)
    ax.set_yscale('log')
    ax.set_xlabel('$\\rho/\\rho_{0}$', fontsize=30)
    ax.set_ylabel('Pressure (MeV/fm3)', fontsize=30)
    return ax

def PlotSkyrmePressureEnergy(df, ax, range_=[0,3], pfrac=0, **args):
    n = np.linspace(*range_, num=1000)
    for index, row in df.iterrows():
        energy = (sky.GetEnergy(n, pfrac, row) + mn)*(n)
        pressure = sky.GetAutoGradPressure(n, pfrac, row)
        labels = ax.plot(energy, pressure, label=index, zorder=1, **args)
    ax.set_ylim([0.5,1e4])
    ax.set_xlim([1e2, 3e4])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('$Energy\ density\ (MeV/fm^{3})$', fontsize=30)
    ax.set_ylabel('$Pressure (MeV/fm^{3})$', fontsize=30)
    return labels

def PlotMassVsRadius(mass, radius, ax, **args):
    for key in mass:
        ax.plot(radius[key], mass[key], label=key, **args)
    ax.set_xlabel('Radius (km)', fontsize=30)
    ax.set_ylabel('Mass (solar)', fontsize=30)
    ax.set_ylim([0, 3])
    ax.set_xlim([0, 20])
    return ax

def PlotMassVsLambda(mass, lambda_, ax, **args):
    for key in mass:
        ax.plot(mass[key], lambda_[key], label=key, **args)
    ax.set_xlabel('Mass (solar)', fontsize=30)
    ax.set_ylabel('lambda', fontsize=30)
    ax.set_ylim([0, 4000])
    return ax

def PlotLambdaRadius(mass, radius, lambda_, ax, **args):
    for key in mass:
        ax.plot([np.interp(1.4, mass[key], radius[key])], [np.interp(1.4, mass[key], lambda_[key])], label=key, marker=marker.next(), markersize=20, **args)
    ax.set_xlabel('R(1.4 solar mass) km', fontsize=30)
    ax.set_ylabel('lambda', fontsize=30)
    ax.set_ylim([-200, 2000])
    ax.set_xlim([0, 20])
    return ax
    
