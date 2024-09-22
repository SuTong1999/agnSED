import numpy as np
from .setUp import *
from .utils import *
from .agnSED import *
from .photometry import *

Mathis_lambda = np.array([0.091, 0.10, 0.13, 0.143, 0.18, 0.20, 0.21, 0.216, 0.23, 0.25,\
    0.346, 0.435, 0.55, 0.7, 0.9, 1.2, 1.8, 2.2, 2.4, 3.4,\
    4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 20.0, 25.0, 30.0, 40.0,\
    50.0, 60.0, 70.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0,\
    800.0, 1000.0])
MathisAv = np.array([5.720, 4.650, 2.960, 2.700, 2.490, 2.780, 3.000, 3.120, 2.860, 2.350,\
    1.580, 1.320, 1.000, 0.750, 0.480, 0.280, 0.160, 0.122, 0.093, 0.038,\
    0.024, 0.018, 0.014, 0.013, 0.072, 0.030, 0.065, 0.062, 0.032, 0.017,\
    0.014, 0.012, 9.7e-3, 8.5e-3, 6.5e-3, 3.7e-3, 2.5e-3, 1.1e-3, 6.7e-4, 2.5e-4,\
    1.4e-4, 7.3e-5])
MathisAlbedo = np.array([0.42, 0.43, 0.45, 0.45, 0.53, 0.56, 0.56, 0.56, 0.63, 0.63,\
    0.71, 0.67, 0.63, 0.56, 0.50, 0.37, 0.25, 0.22, 0.15, 0.058,\
    0.046, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,\
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,\
    0.00, 0.00])

def Mathis_Av(wave):
    wave_um = wave/1e4    # in um
    ind = len(np.argwhere(wave_um>=Mathis_lambda).reshape(-1))-1
    x1 = Mathis_lambda[ind]
    x2 = Mathis_lambda[ind+1]
    y1 = MathisAv[ind]
    y2 = MathisAv[ind+1]
    z1 = MathisAlbedo[ind]
    z2 = MathisAlbedo[ind+1]
    
    Av = (y2-y1)/(x2-x1) * (wave_um-x1) + y1
    albedo = (z2-z1)/(x2-x1) * (wave_um-x1) + z1
    return Av, albedo


def magLgal(sample, wave, component=True, mcut=4, mdotcut=-12, npro=1):
    """ calculate the AB magnitude at a given frequency, the dust extinction model is the same as in Lgalaxies """
    """ inputs: sample: n*4 numpy array, logarithm of BH mass, accretion rate, hydrogen density, metallicity,
                         [[logm1, logmdot1, logNH1, logZ1] , ...]
                wave: wavelength in A
                mcut: BH mass lower limit, if logm<mcut, skip calculation
                mdotcut: BHAR lower limit
                npro: number of cores to multiprocess """  

    lognu0 = np.log10(3e8/(wave*1e-10))
    ind = np.argwhere( (sample[:,0]>mcut)&(sample[:,1]>mdotcut) ).reshape(-1)
    sample = sample[ind]
    logNH = sample[:,2]
    Z_Zsun = 10**sample[:,3] * 0.02/0.01524

    A_Av, albedo = Mathis_Av(wave)
    if (wave<2000.) : s=1.35
    else: s=1.6

    tau_ISM = A_Av * Z_Zsun**s * np.sqrt((1.-albedo)) * 10**(logNH-21)/2.1
    cos_in = np.random.uniform(low=0., high=1.0, size=len(logNH)) 
    cos_in[cos_in<=0.2] = 0.2
    tau_sec = tau_ISM / cos_in
    # if (tau_sec<=0).any(): print('unrealistic tau')
    A_lambda = -2.5 * np.log10((1.-np.exp(-tau_sec))/tau_sec)   # extinction in magnitude

    if component:
        ind_adaf = sample[:,1]<=logmdot_crit
        ind_diskcor = sample[:,1]>logmdot_crit
        logLnu_adaf, logLnu_diskcor = LuminosityPerHerz(sample[:,:2], lognu0, mcut=mcut, mdotcut=mdotcut, npro=npro)
        mag_adaf = -2.5 * (logLnu_adaf + np.log10(1. / 4. / np.pi / (3.08e19)**2)) - 48.6
        mag_diskcor = -2.5 * (logLnu_diskcor + np.log10(1. / 4. / np.pi / (3.08e19)**2)) - 48.6
        mag_adaf_ex = mag_adaf + A_lambda[ind_adaf]
        mag_diskcor_ex = mag_diskcor + A_lambda[ind_diskcor]
        return mag_adaf, mag_diskcor, mag_adaf_ex, mag_diskcor_ex
    else:
        logLnu = LuminosityPerHerz(sample[:,:2], lognu0, component=False, mcut=mcut, mdotcut=mdotcut, npro=npro)
        mag = -2.5 * (logLnu + np.log10(1. / 4. / np.pi / (3.08e19)**2)) - 48.6
        mag_ex = mag + A_lambda
        return mag, mag_ex

