from pptx import Presentation
from pptx.util import Inches
import Image
import matplotlib.pyplot as plt

from PlotEOS3 import EOSDrawer
from MakeMovie import CreateGif
from MakeSkyrmeFileBisection import LoadSkyrmeFile
# Calculate Polarizability to begin with
#df = LoadSkyrmeFile('SkyrmeParameters/PawelSkyrme.csv')
#df = CalculatePolarizability(df, PRCTransDensity=0.1, PolyTropeDensity=2.5*rho0)
#df = AddPressure(df)
#df = AddCausailty(df)
df = LoadSkyrmeFile('test.csv')



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
drawer.DrawEOS(df_causal, ax=ax, xlim=[1e-4, 1e4], ylim=[1e-4, 1e4], color=['b']*6)
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
ax.plot(df_causal['R(1.4)'], df_causal['lambda(1.4)'], 'ro', label='Causal', color='b')
ax.plot(df_acausal['R(1.4)'], df_acausal['lambda(1.4)'], 'ro', label='Acausal', color='r')
ax.set_xlabel('Neutron Star Radius (km)')
ax.set_ylabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
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
ax.plot(df_causal['lambda(1.4)'], df_causal['P(2rho0)'], 'ro', label='Causal', color='b')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['P(2rho0)'], 'ro', label='Acausal', color='r')
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$P(2\rho_{0})$')
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
ax.plot(df_causal['lambda(1.4)'], df_causal['P(1.5rho0)'], 'ro', label='Causal', color='b')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['P(1.5rho0)'], 'ro', label='Acausal', color='r')
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$P(1.5\rho_{0})$')
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
ax.plot(df_causal['lambda(1.4)'], df_causal['P(0.67rho0)'], 'ro', label='Causal', color='b')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['P(0.67rho0)'], 'ro', label='Acausal', color='r')
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$P(0.67\rho_{0})$')
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
ax.plot(df_causal['lambda(1.4)'], df_causal['Sym(2rho0)'], 'ro', label='Causal', color='b')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['Sym(2rho0)'], 'ro', label='Acausal', color='r')
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
ax.plot(df_causal['lambda(1.4)'], df_causal['Sym(1.5rho0)'], 'ro', label='Causal', color='b')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['Sym(1.5rho0)'], 'ro', label='Acausal', color='r')
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
ax.plot(df_causal['lambda(1.4)'], df_causal['Sym(0.67rho0)'], 'ro', label='Causal', color='b')
ax.plot(df_acausal['lambda(1.4)'], df_acausal['Sym(0.67rho0)'], 'ro', label='Acausal', color='r')
ax.set_xlabel(r'$Tidal\ \ Deformability\ \ \Lambda$')
ax.set_ylabel(r'$Sym(0.67\rho_{0})$')
plt.legend()
plt.savefig(figname)
slide.shapes.add_picture(figname, left, top, height=height, width=width)
plt.close()
prs.save('test.pptx')
