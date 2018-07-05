import sys
from pebble import ProcessPool
import scipy.optimize as opt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import Utilities.Utilities as tul
import Utilities.SkyrmeEOS as sky
from Utilities.Constants import *
from EOSCreator import EOSCreator

if __name__ == "__main__":

    df = pd.read_csv('Results/Skyrme_pd_0.7_mm_2.2.csv', index_col=0)
    df.fillna(0, inplace=True)

    def ViolateCausality(eos_name):
        skyrme = sky.Skryme(df.loc[eos_name])
        #skyrme.ToCSV('AllSkyrmes/%s.csv' % eos_name, np.linspace(1e-14, 3*0.16, 100), 0)
        eos_creator = EOSCreator(skyrme, PRCTransDensity=0.7)
        pressure_high = df['PolyHighP'].loc[eos_name]
        eos_creator.PressureHigh = pressure_high
        eos, _ = eos_creator.GetEOSType("EOS")
        

        # Get density corresponding to the high pressure point so we can plot things easier
        max_pressure = df['PCentral2MOdot'].loc[eos_name]
        def GetDensityFromPressure(rho):
            pressure =  eos.GetAutoGradPressure(rho, 0) - max_pressure
            return pressure
        density = opt.newton(GetDensityFromPressure, x0=7*0.16)

        rho = np.linspace(1e-8, density, 1000)
        pressure = eos.GetAutoGradPressure(rho, 0)
        sound = np.array(eos.GetSpeedOfSound(rho, 0))

        if all(sound <= 1):
            print('%s | %r | %10.3f' % (eos_name, False, density/rho0))
            return eos_name, False
        else:
            index = np.argmax(sound > 1)
            den = rho[index]
            print('%s | %r | %10.3f | %10.3f | %10.3f' % (eos_name, True, den/rho0, eos.GetAutoGradPressure(den, 0), density/rho0))
            return eos_name, True

    name_list = [index for index, row in df.iterrows()]
    result = []
    with ProcessPool() as pool:
        future = pool.map(ViolateCausality, name_list)
        iterator = future.result()
        while True:
            try:
                result.append(next(iterator))
            except StopIteration:
                break
            except:
                print sys.exc_info()[0]
                raise
    
    data = [{'Model': val[0], 'ViolateCausality': val[1]} for val in result]
    data = pd.DataFrame.from_dict(data)
    data.set_index('Model', inplace=True)
    df = pd.concat([df, data], axis=1)
    df.to_csv('test.csv')
