import matplotlib.pyplot as plt
import numpy as np
from Utilities.Utilities import PlotMassVsRadius, PlotLambdaRadius, PlotMassVsLambda
import cPickle as pickle

if __name__ == "__main__":
    with open("Results/all_results.pkl", "rb") as buff:\
        data = pickle.load(buff)

    mass = data['mass']
    radius = data['radius']
    lambda_ = data['lambda']
    
    ax = plt.subplot(111)    
    PlotMassVsRadius(data['mass'], data['radius'], ax)
    plt.show()
    ax = plt.subplot(111)
    PlotLambdaRadius(data['mass'], data['radius'], data['lambda'], ax)
    ax.xaxis.set_ticks(np.arange(0.,20.,1.))
    plt.show()
    ax = plt.subplot(111)
    PlotMassVsLambda(data['mass'], data['lambda'], ax)
    plt.show()
    #with open("Lambda.dat", "wb") as file_:
    #    file_.write('ModelName\tR(1.4)\tLambda(1.4)\n')
    #    for key in mass:
            # trying to find the radius and the polarizability at 1.4 solar mass
            # using simple linear interpolation
    #        file_.write('%s\t%f\t%f\n' % (key, np.interp(1.4, mass[key], radius[key]), np.interp(1.4, mass[key], lambda_[key])))
        
