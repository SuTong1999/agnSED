import numpy as np
from multiprocessing import Pool
from .setUp import *
from .utils import *
 

def SED(logm, logmdot, k0=0, k1=len(lognu)-1, truncation=True):
    """calculate the spectral energy distribution (SED)"""
    """use different accretion model for different accretion rate"""
    if logmdot<=logmdot_crit: func = func_adaf
    else: func = func_diskcor
    logLnu = np.array(list(map(lambda f: f([logm, logmdot]), func[k0:k1+1])))
    logLnu[np.isnan(logLnu)] = -100
    if (logmdot>=logmdot_trunc) & (logmdot<=logmdot_crit) & truncation:  
        logLnu_trunc = np.array(list(map(lambda f: f([logm, logmdot]), func_trunc[k0:k1+1])))
        logLnu_trunc[np.isnan(logLnu_trunc)] = -100
        logLnu = np.log10( 10**logLnu + 10**logLnu_trunc )
    return logLnu.reshape(-1)

def SED_SSD(logm, logmdot, k0=0, k1=len(lognu)-1):
    """calculate the spectral energy distribution (SED) of a standard disk"""
    logLnu = np.array(list(map(lambda f: f([logm, logmdot]), func_SSD[k0:k1+1])))
    logLnu[np.isnan(logLnu)] = -100   
    return logLnu.reshape(-1)

def SED_adaf(logm, logmdot, k0=0, k1=len(lognu)-1, truncation=True):
    """calculate the spectral energy distribution (SED) of a ADAF (+truncated disk)"""
    logLnu = np.array(list(map(lambda f: f([logm, logmdot]), func_adaf[k0:k1+1])))
    logLnu[np.isnan(logLnu)] = -100   
    if (logmdot>=logmdot_trunc) & truncation:
        logLnu_trunc = np.array(list(map(lambda f: f([logm, logmdot]), func_trunc[k0:k1+1])))
        logLnu_trunc[np.isnan(logLnu_trunc)] = -100   
        logLnu = np.log10( 10**logLnu+10**logLnu_trunc )
    return logLnu.reshape(-1)

def SED_diskcor(logm, logmdot, k0=0, k1=len(lognu)-1):
    """calculate the spectral energy distribution (SED) of a disk-corona system"""
    logLnu = np.array(list(map(lambda f: f([logm, logmdot]), func_diskcor[k0:k1+1])))
    logLnu[np.isnan(logLnu)] = -100   
    return logLnu.reshape(-1)

def SEDRegimePool(args):
    logm, logmdot, k0, k1 = args
    if logmdot<=logmdot_crit:  # ADAF regime
        logLnu = SED_adaf(logm, logmdot, int(k0), int(k1))
        regime = 0
    else:  # Disk-corona regime
        logLnu = SED_diskcor(logm, logmdot, int(k0), int(k1))
        regime = 1
    return np.array([regime, logLnu], dtype=object)

def SEDPool(args):
    """calculate the spectral energy distribution (SED)"""
    """use different accretion model for different accretion rate"""
    logm, logmdot, k0, k1 = args
    k0 = int(k0)
    k1 = int(k1)
    if logmdot<=logmdot_crit: func = func_adaf
    else: func = func_diskcor
    logLnu = np.array(list(map(lambda f: f([logm, logmdot]), func[k0:k1+1])))
    logLnu[np.isnan(logLnu)] = -100
    if (logmdot>=logmdot_trunc) & (logmdot<=logmdot_crit):  
        logLnu_trunc = np.array(list(map(lambda f: f([logm, logmdot]), func_trunc[k0:k1+1])))
        logLnu_trunc[np.isnan(logLnu_trunc)] = -100
        logLnu = np.log10( 10**logLnu + 10**logLnu_trunc )
    return logLnu.reshape(-1)

def SEDs(sample, lognu0=lognu[0], lognu1=lognu[-1], component=True, mcut=4, mdotcut=-12, npro=1):
    """calculate the AGN SED for an array of BH mass and accretion rate pairs"""
    """inputs: sample: n*2 numpy array, logarithm of BH mass and accretion rate. [[logm1, logmdot1], [logm2, logmdot2], ...]
                [lognu0, lognu1]: edges of frequency band (in logarithm)
                mcut: BH mass lower limit, if logm<mcut, skip calculation
                mdotcut: BHAR lower limit
                npro: number of cores to multiprocess"""
    k0 = np.sum(lognu<=lognu0) -1
    k1 = len(lognu) - np.sum(lognu>=lognu1)
    lognu_band = np.zeros(k1-k0+1)
    lognu_band[0] = lognu0
    lognu_band[-1]= lognu1
    lognu_band[1:-1] = lognu[k0+1:k1]
    ind = np.argwhere( (sample[:,0]>mcut)&(sample[:,1]>mdotcut) ).reshape(-1)
    sample = sample[ind]
    logm = sample[:,0]
    logmdot = sample[:,1]
    data = zip(logm, logmdot, np.ones(len(logm))*k0, np.ones(len(logm))*k1)
    if component:
        with Pool(processes=npro) as pool:
            results = pool.map(SEDRegimePool, data)
        results = np.array(results,dtype=object)
        logLnu_adaf = np.vstack(results[results[:,0]==0, 1])
        logLnu_diskcor = np.vstack(results[results[:,0]==1, 1])
        logLnu_adaf[:,0] = linear_interp(lognu0, lognu[k0], lognu[k0+1], logLnu_adaf[:,0], logLnu_adaf[:,1])
        logLnu_adaf[:,-1] = linear_interp(lognu1, lognu[k1-1], lognu[k1], logLnu_adaf[:,-2], logLnu_adaf[:,-1])
        logLnu_diskcor[:,0] = linear_interp(lognu0, lognu[k0], lognu[k0+1], logLnu_diskcor[:,0], logLnu_diskcor[:,1])
        logLnu_diskcor[:,-1] = linear_interp(lognu1, lognu[k1-1], lognu[k1], logLnu_diskcor[:,-2], logLnu_diskcor[:,-1])
        return logLnu_adaf, logLnu_diskcor
    else:
        with Pool(processes=npro) as pool:
            logLnu = pool.map(SEDPool, data)
        logLnu = np.array(logLnu)
        logLnu[:,0] = linear_interp(lognu0, lognu[k0], lognu[k0+1], logLnu[:,0], logLnu[:,1])
        logLnu[:,-1] = linear_interp(lognu1, lognu[k1-1], lognu[k1], logLnu[:,-2], logLnu[:,-1])
        return logLnu


