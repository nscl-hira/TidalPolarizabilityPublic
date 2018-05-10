from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd

import Utilities.Utilities as utl
import Utilities.SkyrmeEOS as sky 
from Utilities.Constants import *

# obs -- observed value
# exp -- expected value

def chisqr(obs, exp, error):
    return (np.power((obs-exp),2)/(np.power(error, 2))).sum()

def SelectLowDensity(constraint_filename, df):
    """
    Input: 
       constraint_filename: filename containing all lower energy constraint points
       df: list of skyrme to be selected from
    Output:
       List of selected skyrme
       density of constraint points
       Sym energy of constraint points
       Error in density
       Error in Sym energy
    """
    constraints = pd.read_csv(constraint_filename)
    num_constraints = constraints.shape[0]

    delete_model = []
    for index, row in df.iterrows():
        asym = sky.GetAsymEnergy(constraints['rho']*rho0, row)
        chi_square = chisqr(asym, constraints['S'], constraints['S_Error'])
        # only accept models with chisqr per deg. freedom < 3
        if chi_square/float(num_constraints) > 2:
            delete_model.append(index)

    return df.drop(delete_model), constraints['rho'], constraints['S'], \
           constraints['rho_Error'], constraints['S_Error']
    

if __name__ == "__main__":
    df = pd.read_csv('SkyrmeParameters/PawelSkyrme.csv', index_col=0)
    df.fillna(0, inplace=True)
 
    constrainted_df, rho, S, rho_Error, S_Error = SelectLowDensity('Constraints/LowEnergySym.csv', df)
    # save result to a file first
    constrainted_df.to_csv('SkyrmeParameters/SkyrmeConstraintedLowDensity.csv', sep=',')
    
    ax = plt.subplot(121)
    # also plot all for comparison for symmetry term
    ax = utl.PlotSkyrmeSymEnergy(df, ax, color='b', range_=[0,5])
    ax = utl.PlotSkyrmeSymEnergy(constrainted_df, ax, color='r', range_=[0,5])
    ax.errorbar(rho, S, xerr=rho_Error, yerr=S_Error, 
                fmt='o', color='black', ecolor='black', 
                markersize=10, elinewidth=2)
    ax.set_ylim([0,500])
    ax.set_xlim([0,5])
    

    ax = plt.subplot(122)
    # plot background as comparison
    ax = utl.PlotSkyrmeEnergy(df, ax, color='b', range_=[0,5])
    ax = utl.PlotSkyrmeEnergy(constrainted_df, ax, color='r', range_=[0,5])
    value, contour = utl.GetContour(constrainted_df, 0.3, 1)
    # write contour to file
    np.savetxt('Results/E_Constrainted_with_S.csv', np.array([contour, value]), delimiter=',') 
    ax.plot(value, contour, color='black', linewidth=5)
    ax.set_ylim([0,500])
    ax.set_xlim([0,5])
    plt.show()
