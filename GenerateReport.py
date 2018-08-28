#!/projects/hira/tsangc/Polarizability/myPy/bin/python -W ignore
import argparse
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
import matplotlib.pyplot as plt

from Utilities.EOSDrawer import EOSDrawer
from Utilities.MakeMovie import CreateGif
from MakeSkyrmeFileBisection import LoadSkyrmeFile, CalculatePolarizability
from SelectPressure import AddPressure
from SelectSpeedOfSound import AddCausailty
from Utilities.Constants import *

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--Input", default="SkyrmeParameters/PawelSkyrmeNew.csv", help="Name of the Skyrme input file (Default: SkyrmeResult/PawelSkyrmeNew.csv)")
parser.add_argument("-o", "--Output", default="Result", help="Name of the CSV output (Default: Result)")
parser.add_argument("-et", "--EOSType", default="EOS", help="Type of EOS. It can be: EOS, EOSNoPolyTrope, BESkyrme, OnlySkyrme (Default: EOS)")
parser.add_argument("-sd", "--SkyrmeDensity", type=float, default=0.3, help="Density at which Skyrme takes over from crustal EOS (Default: 0.3)")
parser.add_argument("-pp", "--PolyTropeDensity", type=float, default=3, help="Density at which Skyrme EOS ends. (Default: 3)")
parser.add_argument("-td", "--TranDensity", type=float, default=0.001472, help="Density at which Crustal EOS ends (Default: 0.001472)")
parser.add_argument("-pd", "--PRCTransDensity", type=float, default=-1, help="Enable PRC automatic density transition. Value entered determine fraction of density that is represented by relativistic gas")
parser.add_argument("-cs", "--CrustSmooth", type=float, default=0, help="degrees of smoothing. Reduce oscillation of speed of sound near crustal volumn")
parser.add_argument("-mm", "--MaxMassRequested", type=float, default=2, help="Maximum Mass to be achieved for EOS in unit of solar mass (Default: 2)")
args = parser.parse_args()

df = LoadSkyrmeFile(args.Input)
argd = vars(args)
rho0 = 0.16
argd['TranDensity'] = argd['TranDensity']*rho0
argd['SkyrmeDensity'] = argd['SkyrmeDensity']*rho0
argd['PolyTropeDensity'] = argd['PolyTropeDensity']*rho0

# Calculate Polarizability to begin with
df = CalculatePolarizability(df, **argd)
df = AddPressure(df)
df = AddCausailty(df)
#df = LoadSkyrmeFile('test.csv')

prs = Presentation()
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'EOS Pressure vs Energy density'

left = Inches(1.62)
top = Inches(1.55)
height = Inches(5.5)
width = Inches(6.77)

# Plot all the EOSs
figname = 'Report/EOSSection.png'
drawer = EOSDrawer(df)
ax = plt.subplot(111)
drawer.DrawEOS(ax=ax, xlim=[1e-2, 1e4], ylim=[1e-4, 1e4])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$Energy\ Density\ (MeV\ fm^{-3})$')
ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
plt.savefig(figname) 
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

figname = 'Report/EOSPressure.png'
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'EOS Pressure vs rho' 
drawer.DrawEOS(ax=ax, xname='rho', yname='GetAutoGradPressure', xlim=[1e-8, 10*0.16], ylim=[1e-4, 1e4])
ax.set_yscale('log')
ax.set_xlabel(r'$\rho\ fm^{-3}$')
ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
#plt.show()
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

figname = 'Report/EOSEnergyDensity.png'
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'EOS Pressure vs rho' 
drawer.DrawEOS(ax=ax, xname='rho', yname='GetEnergyDensity', xlim=[1e-8, 10*0.16], ylim=[1e-2, 1e4])
ax.set_yscale('log')
ax.set_xlabel(r'$\rho\ fm^{-3}$')
ax.set_ylabel(r'$Energy\ density\ (MeV\ fm^{-3})$')
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

figname = 'Report/EOSCausality.png'
df_causal = df.loc[df['ViolateCausality']==False]
df_acausal = df.loc[df['ViolateCausality']==True]
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'EOS Causal (blue) Acausal (red)'
drawer.DrawEOS(df_causal, ax=ax, xlim=[1e-4, 1e4], ylim=[1e-4, 1e4], color=['g']*6)
drawer.DrawEOS(df_acausal, ax=ax, xlim=[1e-4, 1e4], ylim=[1e-4, 1e4], color=['r']*6)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$Energy\ Density\ (MeV\ fm^{-3})$')
ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
plt.savefig(figname) 
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

figname = 'Report/lambda_radius.png'
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'Lambda vs radius'
ax.plot(df_causal['R(1.4)'], df_causal['lambda(1.4)'], 'ro', label='Causal', color='g')
ax.plot(df_acausal['R(1.4)'], df_acausal['lambda(1.4)'], 'ro', label='Acausal', color='r')
ax.set_xlabel('Neutron Star Radius (km)')
ax.set_ylabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_xlim([9, 16])
ax.set_ylim([-25, 1500])
plt.legend()
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

figname = 'Report/pressure_lambda.png'
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'Pressure (2rho0) vs Lambda'
ax.plot(df_causal['lambda(1.4)'], df_causal['P(2rho0)'], 'ro', label='Causal', color='g')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['P(2rho0)'], 'ro', label='Acausal', color='r')
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$P(2\rho_{0})$')
#ax.set_xscale('log')
ax.set_xlim([0, 1600])
ax.set_ylim([-20, 70])
plt.legend()
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

figname = 'Report/pressure1.5_lambda.png'
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'Pressure (1.5rho0) vs Lambda'
ax.plot(df_causal['lambda(1.4)'], df_causal['P(1.5rho0)'], 'ro', label='Causal', color='g')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['P(1.5rho0)'], 'ro', label='Acausal', color='r')
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$P(1.5\rho_{0})$')
#ax.set_xscale('log')
ax.set_xlim([0, 1600])
ax.set_ylim([-20, 30])
plt.legend()
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

figname = 'Report/pressure0.67_lambda.png'
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'Pressure (0.67rho0) vs Lambda'
ax.plot(df_causal['lambda(1.4)'], df_causal['P(0.67rho0)'], 'ro', label='Causal', color='g')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['P(0.67rho0)'], 'ro', label='Acausal', color='r')
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$P(0.67\rho_{0})$')
#ax.set_xscale('log')
ax.set_xlim([0, 1600])
ax.set_ylim([-1.1, 3.2])
plt.legend()
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

"""
CreateGif([df_causal, df_acausal], 'Report/Sym.gif')
CreateGif([df_causal, df_acausal], 'Report/Pressure.gif', 'GetAutoGradPressure', 0, 50)
prs.save('test.pptx')
"""

figname = 'Report/sym_lambda.png'
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'Sym Term (2rho0) vs Lambda'
ax.plot(df_causal['lambda(1.4)'], df_causal['Sym(2rho0)'], 'ro', label='Causal', color='g')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['Sym(2rho0)'], 'ro', label='Acausal', color='r')
#ax.set_xscale('log')
ax.set_xlim([0, 1600])
ax.set_ylim([-26, 120])
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$Sym(2\rho_{0})$')
plt.legend()
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

figname = 'Report/sym1.5_lambda.png'
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'Sym Term (1.5rho0) vs Lambda'
ax.plot(df_causal['lambda(1.4)'], df_causal['Sym(1.5rho0)'], 'ro', label='Causal', color='g')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['Sym(1.5rho0)'], 'ro', label='Acausal', color='r')
#ax.set_xscale('log')
ax.set_xlim([0, 1600])
ax.set_ylim([-5, 85])
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$Sym(1.5\rho_{0})$')
plt.legend()
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()

figname = 'Report/sym0.67_lambda.png'
ax = plt.subplot(111)
title_only_slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(title_only_slide_layout)
shapes = slide.shapes
shapes.title.text = 'Sym Term (0.67rho0) vs Lambda'
ax.plot(df_causal['lambda(1.4)'], df_causal['Sym(0.67rho0)'], 'ro', label='Causal', color='g')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['Sym(0.67rho0)'], 'ro', label='Acausal', color='r')
#ax.set_xscale('log')
ax.set_xlim([0, 1600])
ax.set_ylim([10, 40])
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$Sym(0.67\rho_{0})$')
plt.legend()
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()
prs.save('Report/%s.pptx' % args.Output)
df.to_csv('Results/%s.csv' % args.Output, index=True)
