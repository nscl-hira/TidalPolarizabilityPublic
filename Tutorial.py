from Utilities.EOSCreator import EOSCreator
from Utilities.EOSCreator import FindCrustalTransDensity
from Utilities.EOSCreator import NuclearEOSFactory
from Utilities.BetaEquilibrium import BetaEquilibrium
from TidalLove import TidalLoveWrapper as wrapper
import Utilities.SkyrmeEOS as sky

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  EOSType = 'EOS2Poly'
  SkyrmeBeforeBetaEqualibrium = NuclearEOSFactory(EOSType, {'t0': -1771.37, 't1': 322.432, 't2': -80.6445, 
                                                  't31': 12219, 't32': 0, 't33': 0, 'x0': 0.266859, 
                                                  'x1': -0.461021, 'x2': 1.18713, 'x31': 0.23156, 
                                                  'x32': 0, 'x33': 0, 'sigma1': 0.333333, 'sigma2': 0, 
                                                  'sigma3': 0, 'rho0': 0.159282694})
  SkyrmeAfterBetaEqualibrium, _, _, _, _ = BetaEquilibrium(SkyrmeBeforeBetaEqualibrium, np.linspace(0.01, 10., 100))
  density = np.linspace(0.1, 3, 100)*0.16
  plt.plot(density, SkyrmeAfterBetaEqualibrium.GetPressure(density))
  plt.show()
 
  # density at which green transition to blue
  crustEndDensity = FindCrustalTransDensity(SkyrmeBeforeBetaEqualibrium)
  # density at which yellow transition to green
  connectStartDensity = crustEndDensity*0.3

  creator = EOSCreator()  
  crustEOS = creator.ConstructCrust('Constraints/EOSCrustOutput.dat', CrustSmooth=0.)

  creator.InsertEOS(crustEOS, connectStartDensity)
  creator.InsertConnection(crustEndDensity)
  creator.InsertEOS(SkyrmeAfterBetaEqualibrium, 30*0.16)#3*SkyrmeBeforeBetaEqualibrium.rho0)
  creator.InsertEOS(lambda prev_density, prev_eos, next_density, next_eos:
                    sky.ConstSpeed.MatchBothEnds(prev_density,
                                                 prev_eos,
                                                 next_density,
                                                 next_eos), 30)
  eosFinal = creator.Build()

  with wrapper.TidalLoveWrapper(eosFinal) as tidal_love:
    result = tidal_love.FindMass(1.4)
    print(result.ToDict())
    result = tidal_love.FindMaxMass()
    print(result.ToDict())
  
