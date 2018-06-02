from shutil import copyfile
import sys
import cPickle as pickle
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
import tempfile
import TidalLove.TidalLove as tidal
from decimal import Decimal
import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from EOSCreator import EOSCreator

if __name__ == "__main__":
    df = pd.read_csv('SkyrmeParameters/PawelSkyrme.csv', index_col=0)
    df.fillna(0, inplace=True)
    
    summary = sky.SummarizeSkyrme(df)

    ax = plt.subplot(121)
    plt.subplots_adjust(right=0.85)
    utl.PlotSkyrmeEnergy(df, ax, color='r')
    ax = plt.subplot(122)
    utl.PlotSkyrmePressure(df, ax, color='r')
    #plt.show()
    
    """
    Print the selected EOS into a file for the tidallove script to run
    """
    def CalculateModel(name_and_eos):
        name = name_and_eos[0]
        eos = EOSCreator(name_and_eos[1])
        with tempfile.NamedTemporaryFile() as output:
            #print header
            output.write(" ========================================================\n")
            output.write("       E/V           P              n           eps      \n") 
            output.write("    (MeV/fm3)     (MeV/fm3)      (#/fm3)    (erg/cm^3/s) \n")
            output.write(" ========================================================\n")
            # the last 2 column (n and eps) is actually not used in the program
            # therefore eps column will always be zero
            n =  np.concatenate([np.linspace(1e-12, 3.76e-4, 1000), np.linspace(3.77e-4, 2, 9000)])
            energy = eos.GetEnergyDensity(n, 0.)
            pressure = eos.GetAutoGradPressure(n, 0.) 
            for density, e, p in zip(n, energy, pressure):
                output.write("   %.5e   %.5e   %.5e   0.0000e+0\n" % (Decimal(e), Decimal(p), Decimal(density)))
            mass, radius, lambda_ = tidal.tidallove(output.name)
            return name, mass, radius, lambda_

    eos_list = [ (index, sky.Skryme(row)) for index, row in df.iterrows() ] 
    eos_list = eos_list[0:20]
    result = []
    num_requested = float(df.shape[0])
    num_completed = 0.
    with ProcessPool() as pool:
        future = pool.map(CalculateModel, eos_list, timeout=100)
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
            num_completed=1+num_completed
            sys.stdout.write('\rProgress Main %f %%' % (100.*num_completed/num_requested))
            sys.stdout.flush()

    mass = {val[0]: val[1] for val in result}
    radius = {val[0]: val[2] for val in result}
    lambda_ = {val[0]: val[3] for val in result}

    data = [{'Model':val[0], 'Max_mass':np.amax(val[1]), 'R(1.4)':np.interp(1.4, val[1], val[2]), 'lambda(1.4)':np.interp(1.4, val[1], val[3]), "1.4PressureC":np.interp(1.4, val[1], np.linspace(3e-5, 3e-5+2e-5*1000, 1000))} for val in result]
    data = pd.DataFrame.from_dict(data)
    print(data)
    data.set_index('Model', inplace=True)
    summary = pd.concat([summary, data], axis=1)
    summary.to_csv('Results/Skyrme_summary_test.csv', index=True)

    ax = plt.subplot(221)
    utl.PlotMassVsRadius(mass, radius, ax, color='b')
    ax = plt.subplot(222)
    utl.PlotMassVsLambda(mass, lambda_, ax, color='b')
    ax = plt.subplot(223)
    utl.PlotLambdaRadius(mass, radius, lambda_, ax, color='b')
    plt.show()

    # save everything into a pickle file
    all_results = {'mass':mass, 'radius':radius, 'lambda':lambda_, 'summary':summary}
    pickle.dump(all_results, open("Results/all_results.pkl", "wb"))

