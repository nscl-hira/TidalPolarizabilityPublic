import sys
import cPickle as pickle
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*')) 
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
import tempfile
from decimal import Decimal
import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd
import scipy.optimize as opt

import TidalLove.TidalLoveWrapper as wrapper
import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from Test import EOSCreator

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('To use, enter: python %s NameOfOutput' % sys.argv[0])
        sys.exit()

    df = pd.read_csv('SkyrmeParameters/PawelSkyrme.csv', index_col=0)
    df.fillna(0, inplace=True)

    summary = sky.SummarizeSkyrme(df)
    
    """
    Print the selected EOS into a file for the tidallove script to run
    """
    def CalculateModel(name_and_eos):
        name = name_and_eos[0]
        eos = EOSCreator(name_and_eos[1])
        tidal_love = wrapper.TidalLoveWrapper(eos)

        def FuncBisec(pc):
            mass, _, _ = tidal_love.Calculate(pc)
            return -mass + 1.4
        try:
            pc = opt.newton(FuncBisec, 1e-3)
            mass, radius, lambda_ = tidal_love.Calculate(pc)
        except RuntimeError as error:
            mass, radius, lambda_ = np.nan, np.nan, np.nan

        # try finding the maximum mass
        try:
            pc = opt.fmin(FuncBisec, 1e-3, disp=False)
            max_mass, _, _ = tidal_love.Calculate(pc[0])
        except RuntimeError as error:
            max_mass = np.nan
            
        if not all([mass, radius, lambda_, max_mass]):
            mass, radius, lambda_, max_mass = np.nan, np.nan, np.nan, np.nan
        
        print('%s, %f, %f, %f, %f' % (name, mass, radius, lambda_, max_mass))
        return name, mass, radius, lambda_, max_mass

    name_list = []
    num_eos_calculated = 0
    tot_num_eos = df.shape[0]
    for index, row in df.iterrows(): 
        name_list.append((index, sky.Skryme(row)))
        num_eos_calculated = num_eos_calculated + 1
        sys.stdout.write('Percentage %3.0f\r' % (float(num_eos_calculated)/float(tot_num_eos)*100.))
        sys.stdout.flush()

    result = []
    num_requested = float(df.shape[0])
    num_completed = 0.
    with ProcessPool() as pool:
        future = pool.map(CalculateModel, name_list, timeout=60)
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

    data = [{'Model':val[0], 'R(1.4)':val[2], 'lambda(1.4)':val[3], 'MaxMass':val[4]} for val in result]
    data = pd.DataFrame.from_dict(data)
    data.set_index('Model', inplace=True)
    data = pd.concat([df, summary, data], axis=1)
    data.dropna(axis=0, how='any', inplace=True)
    data.to_csv('Results/%s.csv' % sys.argv[1], index=True)

    

