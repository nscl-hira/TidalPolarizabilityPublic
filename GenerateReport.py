#!/projects/hira/tsangc/Polarizability/myPy/bin/python -W ignore
import numpy as np
import argparse
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
import matplotlib.pyplot as plt

from Utilities.EOSDrawer import EOSDrawer
from Utilities.MakeMovie import CreateGif
from MakeSkyrmeFileBisection import LoadSkyrmeFile, CalculatePolarizability
from SelectPressure import AddPressure
from SelectAsym import SelectLowDensity
from SelectSpeedOfSound import AddCausailty
from SelectSymPressure import SelectSymPressure
from Utilities.Constants import *
from Utilities.SkyrmeEOS import Skryme
import GeneratePPTX as pptx

def PressureVsEnergyDensity(drawer, figname, **kwargs):
    ax = plt.subplot(111)
    drawer.DrawEOS(ax=ax, xlim=[1e-2, 1e4], ylim=[1e-4, 1e4], **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Energy\ Density\ (MeV\ fm^{-3})$')
    ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
    plt.savefig(figname) 
    plt.close()

def PressureVsDensity(drawer, figname, **kwargs):
    ax = plt.subplot(111)
    #shapes.title.text = 'EOS Pressure vs rho' 
    drawer.DrawEOS(ax=ax, xname='rho/rho0', yname='GetPressure', xlim=[1e-8, 6], ylim=[1e-4, 1e4], **kwargs)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rho/\rho_{0}$')
    ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
    #plt.show()
    plt.savefig(figname)
    plt.close()    

def RejectedEOS(df, df_orig, figname):
    ax = plt.subplot(111)
    #shapes.title.text = 'EOS Pressure vs rho for EOS not calculated' 
    rho = np.concatenate([np.logspace(np.log(1e-9), np.log(3.76e-4), 100, base=np.exp(1)), np.linspace(3.77e-4, 1.6, 900)])
    for index, row in df_orig.loc[df_orig.index.difference(df.index)].iterrows():#  df_orig[~df.index.values.tolist()]:
        eos = Skryme(row)
        x = eos.GetEnergyDensity(rho, 0)
        y = eos.GetPressure(rho, 0)
        ax.plot(x, y)
    ax.set_xlim([1e-2, 1e4])
    ax.set_ylim([1e-4, 1e4])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Energy\ Density\ (MeV\ fm^{-3})$')
    ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
    #plt.show()
    plt.savefig(figname)
    plt.close()

def EnergyDensityVsDensity(drawer, figname, **kwargs):
    ax = plt.subplot(111)
    drawer.DrawEOS(ax=ax, xname='rho', yname='GetEnergyDensity', xlim=[1e-8, 10*0.16], ylim=[1e-2, 1e4], **kwargs)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rho\ fm^{-3}$')
    ax.set_ylabel(r'$Energy\ density\ (MeV\ fm^{-3})$')
    plt.savefig(figname)
    plt.close()

def Causality(drawer, df_causal, df_causal_sat_asym, df_acausal, figname):
    ax = plt.subplot(111)
    #shapes.title.text = 'EOS Causal (blue) Acausal (red) satisfy low density asym (black)'
    drawer.DrawEOS(df_causal, ax=ax, xlim=[1e-4, 1e4], ylim=[1e-4, 1e4], color=['g']*6)
    drawer.DrawEOS(df_causal_sat_asym, ax=ax, xlim=[1e-4, 1e4], ylim=[1e-4, 1e4], color=['b']*6)
    drawer.DrawEOS(df_acausal, ax=ax, xlim=[1e-4, 1e4], ylim=[1e-4, 1e4], color=['r']*6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Energy\ Density\ (MeV\ fm^{-3})$')
    ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
    plt.savefig(figname) 
    plt.close()

def LambdaVsRadius(drawer, df_causal, df_causal_sat_asym, df_acausal, figname):
    ax = plt.subplot(111)
    #shapes.title.text = 'Lambda vs radius'
    ax.plot(df_causal['R(1.4)'], df_causal['lambda(1.4)'], 'ro', label='Causal', color='g')
    ax.plot(df_acausal['R(1.4)'], df_acausal['lambda(1.4)'], 'ro', label='Acausal', color='r')
    ax.plot(df_causal_sat_asym['R(1.4)'], df_causal_sat_asym['lambda(1.4)'], 'ro', label='Satisfy Asym', color='b')
    ax.set_xlabel('Neutron Star Radius (km)')
    ax.set_ylabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
    ax.set_xlim([7, 16])
    ax.set_ylim([-25, 1500])
    plt.legend()
    plt.savefig(figname)
    plt.close()

def PressureVsLambda(density, df_causal, df_causal_sat_asym, df_acausal, figname, ylim=[-20, 70]):
    ax = plt.subplot(111)
    ax.plot(df_causal['lambda(1.4)'], df_causal['P(%grho0)' % density], 'ro', label='Causal', color='g')
    ax.plot(df_causal_sat_asym['lambda(1.4)'], df_causal_sat_asym['P(%grho0)' % density], 'ro', label='Satisfy Asym', color='b')
    ax.plot(df_acausal['lambda(1.4)'], df_acausal['P(%grho0)' % density], 'ro', label='Acausal', color='r')
    ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
    ax.set_ylabel(r'$P(%g\rho_{0})$' % density)
    #ax.set_xscale('log')
    ax.set_xlim([0, 1600])
    ax.set_ylim(ylim)
    plt.legend()
    plt.savefig(figname)
    plt.close()

def SymVsLambda(density, df_causal, df_causal_sat_asym, df_acausal, figname, ylim=[-26, 120]):
    ax = plt.subplot(111)
    ax.plot(df_causal['lambda(1.4)'], df_causal['Sym(%grho0)' % density], 'ro', label='Causal', color='g')
    ax.plot(df_causal_sat_asym['lambda(1.4)'], df_causal_sat_asym['Sym(%grho0)' % density], 'ro', label='Satisfy Asym', color='b')
    ax.plot(df_acausal['lambda(1.4)'], df_acausal['Sym(%grho0)' % density], 'ro', label='Acausal', color='r')
    #ax.set_xscale('log')
    ax.set_xlim([0, 1600])
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
    ax.set_ylabel(r'$Sym(%g\rho_{0})$' % density)
    plt.legend()
    plt.savefig(figname)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--Input", default="SkyrmeParameters/PawelSkyrmeNew.csv", help="Name of the Skyrme input file (Default: SkyrmeResult/PawelSkyrmeNew.csv)")
    parser.add_argument("-o", "--Output", default="Result", help="Name of the CSV output (Default: Result)")
    parser.add_argument("-et", "--EOSType", default="EOS", help="Type of EOS. It can be: EOS, EOSNoPolyTrope, BESkyrme, OnlySkyrme (Default: EOS)")
    parser.add_argument("-sd", "--SkyrmeDensity", type=float, default=0.3, help="Density at which Skyrme takes over from crustal EOS (Default: 0.3)")
    parser.add_argument("-pp", "--PolyTropeDensity", type=float, default=3, help="Density at which Skyrme EOS ends. (Default: 3)")
    parser.add_argument("-td", "--TranDensity", type=float, default=0.001472, help="Density at which Crustal EOS ends (Default: 0.001472)")
    parser.add_argument("-pd", "--PRCTransDensity", type=float, default=-1, help="Enable PRC automatic density transition. Value entered determine fraction of density that is represented by relativistic gas")
    parser.add_argument("-cs", "--CrustSmooth", type=float, default=0., help="degrees of smoothing. Reduce oscillation of speed of sound near crustal volumn")
    parser.add_argument("-mm", "--MaxMassRequested", type=float, default=2, help="Maximum Mass to be achieved for EOS in unit of solar mass (Default: 2)")
    parser.add_argument("-cf", "--CrustFileName", default='Constraints/EOSCrustOutput.dat', help="Type of crustal EoS used (Default: Constraints/EOSCrustOutput.dat)")
    parser.add_argument("--PBar", dest='PBar', action='store_true', help="Enable if you don't need to display everything during calculation, just a progress bar")
    parser.add_argument("-tg", "--TargetMass", type=float, nargs='+', default=[1.4], help="Target mass of the neutron star. (Default: 1.4)")
    args = parser.parse_args()
    
    df_orig = LoadSkyrmeFile(args.Input)
    argd = vars(args)
    rho0 = 0.16
    argd['TranDensity'] = argd['TranDensity']*rho0
    argd['SkyrmeDensity'] = argd['SkyrmeDensity']*rho0
    argd['PolyTropeDensity'] = argd['PolyTropeDensity']*rho0
    
    # Calculate Polarizability to begin with
    df = CalculatePolarizability(df_orig, **argd)
    drawer = EOSDrawer(df)
     # calculate additional EOS properties
    df = AddPressure(df)
    df = AddCausailty(df)
    df, _ = SelectLowDensity('Constraints/LowEnergySym.csv', df)
    df, _ = SelectSymPressure('Constraints/FlowSymMat.csv', df)
    #df = LoadSkyrmeFile('test.csv')
    df.to_csv('Results/%s.csv' % args.Output, index=True)
    
    df_causal = df.loc[df['ViolateCausality']==False]
    df_acausal = df.loc[df['ViolateCausality']==True]
    df_causal_sat_asym = df_causal.loc[df_causal['AgreeLowDensity']==True]
    
    pars = pptx.CreateFirstSlide('EOS NS simulation', '')
    
    # Plot all the EOSs
    figname = 'Report/EOSSection.png'
    PressureVsEnergyDensity(drawer, figname)
    pptx.ImageOnlySlide(pars, 'Pressure vs energy density for all EoSs', figname)
    
    figname = 'Report/RejectedEOS.png'
    RejectedEOS(df, df_orig, figname)
    pptx.ImageOnlySlide(pars, 'EOS Pressure vs rho for EOS not calculated', figname)

    figname = 'Report/EOSSectionCausal.png'
    PressureVsEnergyDensity(drawer, figname, df=df_causal)
    pptx.ImageOnlySlide(pars, 'Pressure vs energy density for all causal EoSs', figname) 
    
    figname = 'Report/EOSEnergyDensity.png'
    EnergyDensityVsDensity(drawer, figname, df=df_causal)
    pptx.ImageOnlySlide(pars, 'EOS Energy Density vs rho', figname)

    figname = 'Report/EOSPressure.png'
    PressureVsDensity(drawer, figname, df=df_causal)
    pptx.ImageOnlySlide(pars, 'EOS Pressure vs rho', figname)
    
   
    
    figname = 'Report/EOSCausality.png'
    Causality(drawer, df_causal, df_causal_sat_asym, df_acausal, figname)
    pptx.ImageOnlySlide(pars, 'EOS Causal (blue) Acausal (red) satisfy low density asym (black)', figname)
    
    figname = 'Report/lambda_radius.png'
    LambdaVsRadius(drawer, df_causal, df_causal_sat_asym, df_acausal, figname)
    pptx.ImageOnlySlide(pars, 'Lambda vs radius', figname)
    
    
    
    
    figname = 'Report/pressure_lambda.png'
    PressureVsLambda(2, df_causal, df_causal_sat_asym, df_acausal, figname)
    pptx.ImageOnlySlide(pars, 'Pressure (2rho0) vs Lambda', figname)
    
    figname = 'Report/pressure1.5_lambda.png'
    PressureVsLambda(1.5, df_causal, df_causal_sat_asym, df_acausal, figname, [-20, 30])
    pptx.ImageOnlySlide(pars, 'Pressure (1.5rho0) vs Lambda', figname)
    
    figname = 'Report/pressure0.67_lambda.png'
    PressureVsLambda(0.67, df_causal, df_causal_sat_asym, df_acausal, figname, [-1.1, 3.2])
    pptx.ImageOnlySlide(pars, 'Pressure (0.67rho0) vs Lambda', figname)
    
    
    figname = 'Report/sym_lambda.png'
    SymVsLambda(2, df_causal, df_causal_sat_asym, df_acausal, figname)
    pptx.ImageOnlySlide(pars, 'Sym Term (2rho0) vs Lambda', figname)
    
    figname = 'Report/sym1.5_lambda.png'
    SymVsLambda(1.5, df_causal, df_causal_sat_asym, df_acausal, figname, [-5, 85])
    pptx.ImageOnlySlide(pars, 'Sym Term (1.5rho0) vs Lambda', figname)
    
    figname = 'Report/sym0.67_lambda.png'
    SymVsLambda(0.67, df_causal, df_causal_sat_asym, df_acausal, figname, [10, 40])
    pptx.ImageOnlySlide(pars, 'Sym Term (0.67rho0) vs Lambda', figname)
    
    pars.save('Report/%s.pptx' % args.Output)
    
