import math
import numpy as np
import pandas as pd
import Utilities.EOSCreator as ec
from Utilities.Constants import hbar, mn, E0, K0, pi, pi2

def NucProp2Skyrme(S0, L, efms, fi=None, delta_m=None, E0=E0, K0=K0, rho0=0.16):
    if delta_m is not None:
        #need to convert delta_m = mn-mp into fi
        if 0.5*(np.sqrt(delta_m*delta_m + efms*efms) - delta_m + efms) > 0:
            fi = (efms - np.sqrt(delta_m*delta_m + efms*efms))/(delta_m*efms)
        else:
            fi = (efms + np.sqrt(delta_m*delta_m + efms*efms))/(delta_m*efms)
    


    efmi0 = 0.5*np.power(hbar,2)/mn*np.power(1.5*pi2*rho0, 2./3.)
    
    #f53=0.5*(np.power(1+delta, 5./3.) + np.power(1-delta, 5./3.))
    gsur = 24.5
    gsuriso=-4.99
    
    theta_s = (1./efms-1.)*8*np.power(hbar,2)/(mn*rho0)
    efmv = 1./(1./efms - fi)
    theta_v = (1./efmv-1.)*4*np.power(hbar,2)/(mn*rho0)
    
    grt=0.6*(1./efms-1)*efmi0
    gamma=(K0+1.2*efmi0-10*grt)/(1.8*efmi0-6*grt-9*E0)
    beta=(0.2*efmi0-2./3.*grt-E0)*(gamma+1)/(gamma-1)
    alpha=E0-efmi0-8./3.*grt-beta
    
    csym=-1./24.*np.power(1.5*pi2, 2./3.)*(3*theta_v-2*theta_s)*np.power(rho0, 5./3.)
    bsym=(3*S0-L-1./3.*efmi0+2*csym)/(-3*(gamma-1))
    asym=S0-1./3.*efmi0-bsym-csym
    
    # output
    t0=4./3.*alpha/rho0
    t3=16.*beta/(gamma+1)/np.power(rho0, gamma)
    
    x0=-4*asym/rho0/t0-0.5
    x3=-24*bsym/np.power(rho0, gamma)/t3-0.5
    
    a_sky = 32*gsur/rho0
    b_sky= -32*gsuriso/rho0
    c_sky=-24*csym/np.power(1.5*pi2, 2./3.)/np.power(rho0, 5./3.)
    d_sky=80*grt/(3*np.power(1.5*pi2, 2./3.)*np.power(rho0, 5./3.))
    
    t1=(a_sky+d_sky)/12.
    t2=(3*d_sky-(b_sky-2*c_sky)-6*t1)/6.
    x2=(d_sky-3*t1-5*t2)/(4.*t2)
    x1=(c_sky+5*x2*t2+4*t2)/(3*t1)


    row = {'t0': t0, 't1':t1, 't2':t2, 't31':t3, 't32':0, 't33':0,
           'x0': x0, 'x1':x1, 'x2':x2, 'x31':x3, 'x32':0, 'x33':0,
           'sigma1':gamma-1, 'sigma2':0, 'sigma3':0}
    return row

if __name__ == '__main__':
    rho0 = 0.16
    # test generation of random skyrmes
    efms = 0.75
    delta_m = 0.1

    # order: S0, L, K0, ms, fi
    corr = np.array([[1., -0.1, 0.0, 0.0,0.0], [-0.1, 1., 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0],[0.0,0.0,0.0,0.0,1.0]])
    sd = np.array([2., 15., 20., 0.1, 0.1])
    mean = np.array([32., 60., 230., 0.75, -0.05])

    cov = np.dot(np.diag(sd), np.dot(corr, np.diag(sd)))
    
    all_para = []
    S0, L, K0, efms, fi = np.random.multivariate_normal(mean, cov, size=(200000,1,1)).T
    S0 = S0.flatten()
    L = L.flatten()
    K0 = K0.flatten()
    efms = efms.flatten()
    fi = fi.flatten()
    all_para = NucProp2Skyrme(S0, L, efms, fi, K0=K0)
    df = pd.DataFrame(all_para)
    df.to_csv('test.csv')


    """
    creator = ec.EOSCreator(row)
    creator.ImportEOS(EOSType='EOS')
    eos = creator.ImportedEOS
    print('Supplied L %f, Skyrme L %f' % (L, eos.GetL(rho0)))
    print('Supplied E0 %f, Skyrme E0 %f' % (E0, eos.GetEnergy(rho0, 0.5)-mn))
    print('Supplied K0 %f, Skyrme K0 %f' % (K0, eos.GetK(rho0, 0.5)))
    print('Supplied S0 %f, Skyrme S0 %f' % (S0, eos.GetAsymEnergy(rho0)))
    print('Supplied ms %f, Skyrme ms %f' % (efms, eos.GetMs(rho0)))
    #print('Supplied fi %f, Skyrme fi %f' % (fi, eos.GetFI(rho0)))
    """
