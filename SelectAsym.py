import itertools
color = itertools.cycle(('fuchsia', 'r', 'r', 'b', 'g', 'orange')) 
marker = itertools.cycle(('*','v','^','x','d')) 
import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd
import Utilities.Utilities as utl
from Utilities.EOSCreator import EOSCreator
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

    AdditionalColumn = []
    for index, row in df.iterrows():
        creator = EOSCreator(row)
        creator.ImportEOS(**row)
        eos = creator.ImportedEOS
        asym = eos.GetAsymEnergy(constraints['rho'].values*eos.rho0)
        chi_square = chisqr(asym, constraints['S'], constraints['S_Error'])
        # only accept models with chisqr per deg. freedom < 3
        if chi_square/float(num_constraints) > 2:
            AdditionalColumn.append({'Model': index, 'AgreeLowDensity':False})
        else:
            AdditionalColumn.append({'Model': index, 'AgreeLowDensity':True})
        #else:
        #    eos.ToCSV('AllSkyrmes/LowDensityConstrainted/%s.csv' % index, np.linspace(1e-14, 3*0.16, 100), 0)
    new_column = pd.DataFrame(AdditionalColumn)
    new_column.set_index('Model', inplace=True)
    return pd.concat([df, new_column], axis=1), constraints
    

if __name__ == "__main__":
    df = pd.read_csv('Results/Newest.csv', index_col=0)
    df.fillna(0, inplace=True)
 
    constrainted_df, constraints = SelectLowDensity('Constraints/LowEnergySym.csv', df)
    constrainted_df = constrainted_df[constrainted_df['AgreeLowDensity'] == True]
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
