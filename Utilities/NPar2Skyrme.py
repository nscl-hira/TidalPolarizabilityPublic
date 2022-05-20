import math
import numpy as np
#import Utilities.EOSCreator as ec
import Utilities.SkyrmeEOS as sky


# suppose to be input
S0 = 32.
L = 40.
efms = 0.8
fi = 0.0
#delta = 0.2
#input end

hbar = 197.32
mn = 938.0
rho0 = 0.16
E0 = -16
K0 = 230
pi2 = math.pi*math.pi

efmi0 = 0.5*np.power(hbar,2)/mn*np.power(1.5*pi2*rho0, 2./3.)

#f53=0.5*(np.power(1+delta, 5./3.) + np.power(1-delta, 5./3.))
gsur = 24.5
gsuriso=-4.99

theta_s = (1./efms-1.)*8*np.power(hbar,2)/(mn*rho0)
efmv = 1./(1./efms - fi)
theta_v = (1./efmv-1.)*4*np.power(hbar,2)/(mn*rho0)

eff_mn=1/(1.2/efms-0.2/efmv)
eff_mp=1/(0.8/efms+0.2/efmv)

grt=0.6*(1./efms-1)*efmi0
gamma=(K0+1.2*efmi0-10*grt)/(1.8*efmi0-6*grt-9*E0)
beta=(0.2*efmi0-2./3.*grt-E0)*(gamma+1)/(gamma-1)
alpha=E0-efmi0-8./3.*grt-beta

csym=-1./24.*np.power(1.5*pi2, 2./3.)*(3*theta_v-2*theta_s)*np.power(rho0, 5./3.)
bsym=(3*S0-L-1./3.*efmi0+2*csym)/(-3*(gamma-1))
asym=S0-1./3.*efmi0-bsym-csym

c0=1./(16.*np.power(hbar, 2))*theta_v
d0=1./(16.*np.power(hbar, 2))*(theta_s-2*theta_v)

# output
t0=4./3.*alpha/rho0
t3=16.*beta/(gamma+1)/np.power(rho0, gamma)

x0=-4*asym/rho0/t0-0.5
x3=-24*bsym/np.power(rho0, gamma)/t3-0.5

a_sky = 32*gsur/rho0
b_sky=-32*gsuriso/rho0
c_sky=-24*csym/np.power(1.5*pi2, 2./3.)
d_sky=80*grt/(3*np.power(1.5*pi2, 2./3.)*np.power(rho0, 5./3.))

t1=(a_sky+d_sky)/12.
t2=(3*d_sky-(b_sky-2*c_sky)-6*t1)/6.
x2=(d_sky-3*t1-5*t2)/(4.*t2)
x1=(c_sky+5*x2*t2+4*t2)/(3*t1)


print('t0,t1,t2,t3,x0,x1,x2,x3,gamma')
print('%f,%f,%f,%f,%f,%f,%f,%f,%f' % (t0,t1,t2,t3,x0,x1,x2,x3,gamma))

row = {'t0': t0, 't1':t1, 't2':t2, 't31':t3, 't32':0, 't33':0,
       'x0': x0, 'x1':x1, 'x2':x2, 'x31':x1, 'x32':0, 'x33':0,
       'sigma1':gamma-1, 'sigma2':0, 'sigma3':0}
eos = sky.Skryme(row)
print('Supplied L %f, Skyrme L %f' % (L, eos.GetL(rho0)))
print('Supplied E0 %f, Skyrme E0 %f' % (E0, eos.GetEnergy(rho0, 0.5)-mn))
print('Supplied K0 %f, Skyrme K0 %f' % (K0, eos.GetK(rho0, 0.5)))
print('Supplied S0 %f, Skyrme S0 %f' % (S0, eos.GetAsymEnergy(rho0)))
