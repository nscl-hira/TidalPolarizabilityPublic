import numpy as np
from MakeSkyrmeFile import PlotMassVsRadius, PlotLambdaRadius, PlotMassVsLambda
import cPickle as pickle

if __name__ == "__main__":
    with open("all_results.pkl", "rb") as buff:\
        data = pickle.load(buff)

    mass = data['mass']
    radius = data['radius']
    lambda_ = data['lambda']
    
    PlotMassVsRadius(data['mass'], data['radius'])
    PlotLambdaRadius(data['mass'], data['radius'], data['lambda'])
    PlotMassVsLambda(data['mass'], data['lambda'])
    #with open("Lambda.dat", "wb") as file_:
    #    file_.write('ModelName\tR(1.4)\tLambda(1.4)\n')
    #    for key in mass:
            # trying to find the radius and the polarizability at 1.4 solar mass
            # using simple linear interpolation
    #        file_.write('%s\t%f\t%f\n' % (key, np.interp(1.4, mass[key], radius[key]), np.interp(1.4, mass[key], lambda_[key])))
        
