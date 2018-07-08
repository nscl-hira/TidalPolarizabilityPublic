import matplotlib.pyplot as plt

from PlotEOS3 import EOSDrawer
from MakeSkyrmeFileBisection import LoadSkyrmeFile
# Calculate Polarizability to begin with
#df = LoadSkyrmeFile('SkyrmeParameters/PawelSkyrme.csv')
#df = CalculatePolarizability(df, PRCTransDensity=0.1, PolyTropeDensity=2.5*rho0)
#df = AddPressure(df)
#df = AddCausailty(df)

df = LoadSkyrmeFile('test.csv')
# Plot all the EOSs
drawer = EOSDrawer(df)
ax = plt.subplot(111)
drawer.DrawEOS(ax=ax, xlim=[1e-4, 1e4], ylim=[1e-4, 1e4])
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

ax = plt.subplot(111)
drawer.DrawEOS(ax=ax, xname='rho', yname='GetAutoGradPressure', xlim=[1e-8, 10*0.16], ylim=[1e-4, 1e4])
ax.set_yscale('log')
plt.show()

ax = plt.subplot(111)
drawer.DrawEOS(ax=ax, xname='rho', yname='GetEnergyDensity', xlim=[1e-8, 10*0.16], ylim=[1e-4, 1e4])
ax.set_yscale('log')
plt.show()

ax = plt.subplot(111)
drawer.DrawEOS(df.loc[df['ViolateCausality']==False], ax=ax, xlim=[1e-4, 1e4], ylim=[1e-4, 1e4], color=['b']*6)
drawer.DrawEOS(df.loc[df['ViolateCausality']==True], ax=ax, xlim=[1e-4, 1e4], ylim=[1e-4, 1e4], color=['r']*6)
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

ax = plt.subplot(111)
drawer.DrawEOS(df.loc[df['ViolateCausality']==False], ax=ax, xname='rho', yname='GetEnergyDensity', xlim=[1e-8, 1.6], ylim=[1e-4, 1e4], color=['b']*6)
drawer.DrawEOS(df.loc[df['ViolateCausality']==True], ax=ax, xname='rho', yname='GetEnergyDensity', xlim=[1e-8, 1.6], ylim=[1e-4, 1e4], color=['r']*6)
ax.set_yscale('log')
plt.show()
