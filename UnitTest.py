import pandas as pd
import numpy as np
import unittest

from TidalLove import TidalLoveWrapper as wrapper
from Utilities.EOSCreator import EOSCreator
import UnitTestData as data

class TestEOS(unittest.TestCase):
    def setUp(self):
        self.test_file = pd.read_csv('SkyrmeParameters/UnitTest.csv', index_col=0).fillna(0)
        self.eos_kwargs = self.test_file.iloc[0]
        self.creator = EOSCreator()

    def CompareResult(self, eos, density_list, target_density_list, target_energy, target_pressure):
        energy = eos.GetEnergyDensity(data.density)
        pressure = eos.GetPressure(data.density)

        energy_frac = np.mean(np.absolute((energy - target_energy)/target_energy))
        pressure_frac = np.mean(np.absolute((pressure- target_pressure)/target_pressure))
        trans_dens_L2 = np.sqrt(np.mean(np.square(np.array(density_list) - target_density_list)))

               
        # assert mean difference to less than 1%
        self.assertAlmostEqual(energy_frac, 0, delta=0.01)
        # same for pressure
        self.assertAlmostEqual(pressure_frac, 0, delta=0.01)
        self.assertAlmostEqual(trans_dens_L2, 0, delta=1e-3, msg='Traget density: %s, New density %s' % (str(data.EOSTransDens), str(density_list)))

    def CompareCrustResult(self, eos, density_list, target_density_list, target_energy, target_pressure):
        # find L2 difference between energy density and pressure
        crustal_id = data.density < density_list[1]
        energy = eos.GetEnergyDensity(data.density[crustal_id])
        pressure = eos.GetPressure(data.density[crustal_id])
        target_energy = target_energy[crustal_id] 
        target_pressure = target_pressure[crustal_id]

        # strict test needed for crustal region
        energy_crust_frac = np.mean(np.absolute((energy - target_energy)/target_energy))
        pressure_crust_frac = np.mean(np.absolute((pressure - target_pressure)/target_pressure))
        # allow for 1 percent error
        self.assertAlmostEqual(energy_crust_frac, 0, delta=1e-5)
        self.assertAlmostEqual(pressure_crust_frac, 0, delta=5e-3)

    def CompareDeformability(self, eos, target_max_mass, target_radius, target_lambda):
        with wrapper.TidalLoveWrapper(eos) as tidal_love:
            result = tidal_love.FindMaxMass()
            # allow for 1 percent error
            self.assertAlmostEqual((result.mass - target_max_mass)/target_max_mass, 0, delta=1e-2)
            result = tidal_love.FindMass(1.4)
            self.assertAlmostEqual((result.Radius - target_radius)/target_radius, 0, delta=1e-2)
            self.assertAlmostEqual((result.Lambda - target_lambda)/target_lambda, 0, delta=1e-2)


    def test_EOS(self):
        eos, density_list, _ = self.creator.Factory(EOSType='EOS', 
                                                    Backbone_kwargs=self.eos_kwargs,
                                                    Transform_kwargs=data.EOSTransKwargs)

        with self.subTest(AllResult=1):
            self.CompareResult(eos, density_list, data.EOSTransDens, data.EOSEnergyDensity, data.EOSPressure)
        with self.subTest(CrustResult=1):
            self.CompareCrustResult(eos, density_list, data.EOSTransDens, data.EOSEnergyDensity, data.EOSPressure)
        with self.subTest(MaxMass=1):
            self.CompareDeformability(eos, data.EOSMaxMass['mass'], data.EOS1_4Mass['Radius'], data.EOS1_4Mass['Lambda'])

    def test_EOS2Poly(self):
        eos, density_list, _ = self.creator.Factory(EOSType='EOS2Poly', 
                                                    Backbone_kwargs=self.eos_kwargs,
                                                    Transform_kwargs=data.EOSTransKwargs)
        with self.subTest(AllResult=1):
            self.CompareResult(eos, density_list, data.EOS2PolyTransDens, data.EOS2PolyEnergyDensity, data.EOS2PolyPressure)
        with self.subTest(CrustResult=1):
            self.CompareCrustResult(eos, density_list, data.EOS2PolyTransDens, data.EOS2PolyEnergyDensity, data.EOS2PolyPressure)
        with self.subTest(MaxMass=1):
            self.CompareDeformability(eos, data.EOS2PolyMaxMass['mass'], data.EOS2Poly1_4Mass['Radius'], data.EOS2Poly1_4Mass['Lambda'])

    def test_MetaSound(self):
        eos, density_list, _ = self.creator.Factory(EOSType='MetaSound', 
                                                    Backbone_kwargs=data.MetaKwargs,
                                                    Transform_kwargs=data.MetaTransKwargs)
        with self.subTest(AllResult=1):
            self.CompareResult(eos, density_list, data.MetaTransDens, data.MetaEnergyDensity, data.MetaPressure)
        with self.subTest(CrustResult=1):
            self.CompareCrustResult(eos, density_list, data.MetaTransDens, data.MetaEnergyDensity, data.MetaPressure)
        with self.subTest(MaxMass=1):
            self.CompareDeformability(eos, data.MetaMaxMass['mass'], data.Meta1_4Mass['Radius'], data.Meta1_4Mass['Lambda'])


if __name__ == '__main__':
    unittest.main()
