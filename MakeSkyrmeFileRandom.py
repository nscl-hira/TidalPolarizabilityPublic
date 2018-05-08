import cPickle as pickle
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import tempfile
import tidalpy
from decimal import Decimal
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
import math

# Declare all the constants
hbar     = 197.32
mn       = 938.0
rho0     = 0.16
E0       = -16
K0       = 230
delta    = 0.2
pi       = math.pi
pi2      = math.pi**2

def GetH(n, pfrac):
    return (2**(n-1))*(pfrac**n + (1-pfrac)**n)

def GetEnergy(rho, pfrac, para):
    a = para['t1']*(para['x1']+2) + para['t2']*(para['x2']+2)
    b = 0.5*(para['t2']*(2*para['x2']+1)-para['t1']*(2*para['x1']+1))

    result = 3.*(hbar**2.)/(10.*mn)*((3.*pi2/2.)**0.666667)*np.power(rho, 0.6667)*GetH(5./3., pfrac)
    result += para['t0']/8.*rho*(2.*(para['x0']+2.)-(2.*para['x0']+1)*GetH(2., pfrac))
    for i in xrange(1, 4):
        result += 1./48.*para['t3%d'%i]*(rho**(para['sigma%d'%i]+1.))*(2.*(para['x3%d'%i]+2.)-(2.*para['x3%d'%i]+1.)*GetH(2., pfrac))
    result += 3./40.*((3.*pi2/2.)**0.666667)*np.power(rho, 5./3.)*(a*GetH(5./3., pfrac)+b*GetH(8./3., pfrac))
    return result

def GetAutoGradPressure(rho, pfrac, para):
    def EDensity(rho_):
        return GetEnergy(rho_, pfrac, para)
    grad_edensity = egrad(EDensity)
    return rho*rho*grad_edensity(rho)


def PlotSkyrmeEnergy(df):
    fig = plt.figure()
    ax = plt.subplot(111)
    n = np.linspace(0, 5, 100)
    for index, row in df.iterrows():
        energy = GetEnergy(n*0.16, 0.5, row)
        ax.plot(n, energy, label='%s SNM' % index, marker=marker.next())
    for index, row in df.iterrows():
        energy = GetEnergy(n*0.16, 0., row)
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
        pressure = GetAutoGradPressure(n*0.16, 0.0, row)
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
    

if __name__ == "__main__":
    df = pd.read_csv('PawelSkyrme.csv', index_col=0)
    df.fillna(0, inplace=True)
    samples = 500

    new_dict = {}
    for col in df:
        mean = df[col].mean()
        std = df[col].std()
        new_dict[col] = np.random.uniform(low=mean-2*std, high=mean+2*std, size=(samples,))

    df = pd.DataFrame.from_dict(new_dict)
    print(df.describe())

    
    #PlotSkyrmeEnergy(df)
    #PlotSkyrmePressure(df)
    
    """
    Print the selected EOS into a file for the tidallove script to run
    """
     
    def CalculateModel(name):
        with tempfile.NamedTemporaryFile() as output:
            #print header
            output.write(" ========================================================\n")
            output.write("       E/V           P              n           eps      \n") 
            output.write("    (MeV/fm3)     (MeV/fm3)      (#/fm3)    (erg/cm^3/s) \n")
            output.write(" ========================================================\n")
        
            # the last 2 column (n and eps) is actually not used in the program
            # therefore eps column will always be zero
            n = np.linspace(1e-10, 2, 10000)
            energy = (GetEnergy(n, 0., df.loc[name]) + mn)*n
            pressure = GetAutoGradPressure(n, 0., df.loc[name]) 
            for density, e, p in zip(n, energy, pressure):
                output.write("   %.5e   %.5e   %.5e   0.0000e+0\n" % (Decimal(e), Decimal(p), Decimal(density)))


            mass, radius, lambda_ = tidalpy.tidallove(output.name)
            return name, mass, radius, lambda_

    name_list = [ index for index, row in df.iterrows() ] 
    result = []
    with ProcessPool() as pool:
        future = pool.map(CalculateModel, name_list, timeout=300)
        iterator = future.result()
        while True:
            try:
                result.append(next(iterator))
            except StopIteration:
                break
            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process


    #if pool.isAlive():
    #    print("Calculation is not finished in time. Abort")
    #    pool.terminate()
    #    pool.join()

    mass = {val[0]: val[1] for val in result}
    radius = {val[0]: val[2] for val in result}
    lambda_ = {val[0]: val[3] for val in result}


    PlotMassVsRadius(mass, radius)
    PlotMassVsLambda(mass, lambda_)
    PlotLambdaRadius(mass, radius, lambda_)

    # save everything into a pickle file
    all_results = {'mass':mass, 'radius':radius, 'lambda':lambda_}
    pickle.dump(all_results, open("all_results.pkl", "wb"))

