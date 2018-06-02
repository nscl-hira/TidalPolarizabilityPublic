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

import TidalLove.TidalLoveWrapper as wrapper
import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *
from EOSCreator import EOSCreator

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('To use, enter: python %s NameOfOutput' % sys.argv[0])
        sys.exit()

    df = pd.read_csv('SkyrmeParameters/PawelSkyrme.csv', index_col=0)
    df.fillna(0, inplace=True)

    summary = sky.SummarizeSkyrme(df)

    print('''\
{dashes}
{m:^12} | {r:^12} | {l:^12} | {p:^12} | {a:^12} | {b:^12}
{dashes}'''.format(dashes='-'*100, m='name', r='radius1.4', l='lambda1.4', p='pc1.4', a='max_mass', b='pc'))
    
    """
    Print the selected EOS into a file for the tidallove script to run
    """
    def CalculateModel(name_and_eos):
        name = name_and_eos[0]
        eos = EOSCreator(name_and_eos[1]).GetEOS()
        tidal_love = wrapper.TidalLoveWrapper(eos)
        pc14=0
        pc=[0]

        mass, radius, lambda_, pc14 = tidal_love.FindMass14()
        max_mass, pc = tidal_love.FindMaxMass()
    
        if np.nan in [mass, radius, lambda_, max_mass]:
            mass, radius, lambda_, max_mass = np.nan, np.nan, np.nan, np.nan
        else:
            #print(name, radius, lambda_, pc14, max_mass, pc)
            print("{m:^12} | {r:^12.3f} | {l:^12.3f} | {p:^12.3f} | {a:^12.3f} | {c:^12.3f} ".format(m=name, r=radius, l=lambda_, p=pc14, a=max_mass, c=pc[0]))

        tidal_love.Close()
        
        return name, mass, radius, lambda_, max_mass

    name_list = [(index, sky.Skryme(row)) for index, row in df.iterrows()]

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
                pass
                #print("function took longer than %d seconds" % error.args[1])
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process
            num_completed=1+num_completed
            #sys.stdout.write('\rProgress Main %f %%' % (100.*num_completed/num_requested))
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
