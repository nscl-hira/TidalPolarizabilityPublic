import itertools
color = itertools.cycle(('fuchsia', 'r', 'r', 'b', 'g', 'orange')) 
marker = itertools.cycle(('*','v','^','x','d')) 
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
        eos = sky.Skryme(row)
        asym = eos.GetAsymEnergy(constraints['rho']*rho0)
        chi_square = chisqr(asym, constraints['S'], constraints['S_Error'])
        # only accept models with chisqr per deg. freedom < 3
        if chi_square/float(num_constraints) > 2:
            delete_model.append(index)
        else:
            eos.ToCSV('AllSkyrmes/LowDensityConstrainted/%s.csv' % index, np.linspace(1e-14, 3*0.16, 100), 0)

    return df.drop(delete_model), constraints
    

if __name__ == "__main__":
    df = pd.read_csv('Results/Skyrme_pd_0.7.csv', index_col=0)
    df.fillna(0, inplace=True)
 
    constrainted_df, constraints = SelectLowDensity('Constraints/LowEnergySym.csv', df)
    # save result to a file first
    constrainted_df.to_csv('SkyrmeParameters/SkyrmeConstraintedLowDensity.csv', sep=',')

    ax1, ax2 = utl.PlotMaster(df, [constrainted_df], [None])
    color_list = []
    for index, row in constraints.iterrows():
        color_ = color.next()
        color_list.append(color_)
        ax1.errorbar(row['rho'], row['S'], fmt='ro', xerr=row['rho_Error'], yerr=row['S_Error'], 
                    marker=marker.next(), mfc='w', mec=color_, ecolor=color_, label=('$%s$' % row['label']))
    handles, labels = ax1.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    leg = ax1.legend(handles, labels, loc='upper left', numpoints=1)

    # change the font colors to match the line colors:
    for color_,text in zip(color_list, leg.get_texts()):
        text.set_color(color_)    

    plt.show()
