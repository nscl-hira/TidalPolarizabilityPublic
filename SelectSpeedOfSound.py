import traceback
import sys
from multiprocessing import cpu_count
from pebble import ProcessPool, ProcessExpired
import scipy.optimize as opt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

import Utilities.Utilities as tul
import Utilities.SkyrmeEOS as sky
from Utilities.Constants import *
from Utilities.EOSCreator import EOSCreator
from MakeSkyrmeFileBisection import LoadSkyrmeFile

def ViolateCausality(eos_name, df):
    rho0 = 0.16#skyrme.rho0
    eos_creator = EOSCreator(df.loc[eos_name])
    kwargs = eos_creator.PrepareEOS(**df.loc[eos_name])
    try:
        eos, _ = eos_creator.GetEOSType(**kwargs)
    except ValueError:
        #print('%s | Cannot form EOS' % eos_name)
        return eos_name, True, False

    # Get density corresponding to the high pressure point so we can plot things easier
    max_pressure = df['PCentralMaxMass'].loc[eos_name]
    def GetDensityFromPressure(rho):
        pressure =  eos.GetPressure(rho, 0) - max_pressure
        return pressure
    try:
        density = opt.newton(GetDensityFromPressure, x0=4*0.16)
    except Exception as e:
        density = 7*0.16
    if np.isnan(density):
        density = 7*0.16

    rho = np.concatenate([np.logspace(np.log(1e-9), np.log(3.76e-4), 100, base=np.exp(1)), np.linspace(3.77e-4, density, 900)])
    try:
        pressure = eos.GetPressure(rho, 0)
        sound = np.array(eos.GetSpeedOfSound(rho, 0))
    except Exception:
        return eos_name, True, False

    if all(sound <= 1) and all(sound >=0):
        return eos_name, False, False
    elif any(sound <=0):
        return eos_name, True, True
    else:
        index = np.argmax(sound > 1)
        den = rho[index]
        return eos_name, True, False


def AddCausailty(df, disable=False):
    name_list = [index for index, row in df.iterrows()]
    result = []
    with tqdm(total = len(name_list), ncols=100, disable=disable) as pbar:
        with ProcessPool(max_workers=cpu_count()) as pool:
            future = pool.map(partial(ViolateCausality, df=df), name_list)
            iterator = future.result()
            while True:
                try:
                    result.append(next(iterator))
                    pbar.update(1)
                except StopIteration:
                    break
                except ProcessExpired as error:
                    sys.stderr.write("%s. Exit code: %d" % (error, error.exitcode))
                except Exception as error:
                    pbar.update(1)
                    sys.stderr.write(error.traceback)
    
    data = [{'Model': val[0], 'ViolateCausality': val[1], 'NegSound': val[2]} for val in result]
    data = pd.DataFrame.from_dict(data)
    data.set_index('Model', inplace=True)
    df = pd.concat([df, data], axis=1)
    #df.combine_first(data)
    return df

if __name__ == "__main__":
    df = LoadSkyrmeFile('Results/test.csv')
    #df = df.loc[df['SDToRTDRadius'] < 1e-3]
    df = AddCausailty(df)
    df.to_csv('test.csv')

    
