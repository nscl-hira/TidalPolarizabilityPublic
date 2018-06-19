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

    df = pd.read_csv('Results/Skyrme_pd_0.7.csv', index_col=0)
    df.fillna(0, inplace=True)

    def ViolateCausality(eos_name):
        skyrme = sky.Skryme(df.loc[eos_name])
        #skyrme.ToCSV('AllSkyrmes/%s.csv' % eos_name, np.linspace(1e-14, 3*0.16, 100), 0)
        eos_creator = EOSCreator(skyrme, PRCTransDensity=0.7)
        pressure_high = df['PCentral2MOdot'].loc[eos_name]
        eos_creator.PressureHigh = pressure_high
        eos, _ = eos_creator.GetEOSType("EOS")
        

        # Get density corresponding to the high pressure point so we can plot things easier
        def GetDensityFromPressure(rho):
            pressure =  eos.GetAutoGradPressure(rho, 0) - pressure_high
            return pressure
        density = opt.newton(GetDensityFromPressure, x0=7*0.16)

        rho = np.linspace(1e-14, density, 1000)
        pressure = eos.GetAutoGradPressure(rho, 0)
        sound = eos.GetSpeedOfSound(rho, 0)

        if np.all(sound <= 1):
            print('%s | %r' % (eos_name, False))
            return eos_name, False
        else:
            print('%s | %r' % (eos_name, True))
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
