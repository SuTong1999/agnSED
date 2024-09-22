import numpy as np
from .setUp import *
from .utils import *
 

def SED(logm, logmdot, k0=0, k1=len(lognu)-1, truncation=True):
    """calculate the spectral energy distribution (SED)"""
    """use different accretion model for different accretion rate"""
    if logmdot<=logmdot_crit: func = func_adaf
    else: func = func_diskcor
    logLnu = np.array(list(map(lambda f: f(logmdot, logm), func[k0:k1+1])))
    logLnu[np.isnan(logLnu)] = -100
    if (logmdot>=logmdot_trunc) & (logmdot<=logmdot_crit) & truncation:  
        logLnu_trunc = np.array(list(map(lambda f: f(logmdot, logm), func_trunc[k0:k1+1])))
        logLnu_trunc[np.isnan(logLnu_trunc)] = -100
        logLnu = np.log10( 10**logLnu + 10**logLnu_trunc )
    return logLnu.reshape(-1)

def SED_SSD(logm, logmdot, k0=0, k1=len(lognu)-1):
    """calculate the spectral energy distribution (SED) of a standard disk"""
    logLnu = np.array(list(map(lambda f: f(logmdot, logm), func_wholeSSD[k0:k1+1])))
    logLnu[np.isnan(logLnu)] = -100   
    return logLnu.reshape(-1)

def SED_adaf(logm, logmdot, k0=0, k1=len(lognu)-1, truncation=True):
    """calculate the spectral energy distribution (SED) of a ADAF (+truncated disk)"""
    logLnu = np.array(list(map(lambda f: f(logmdot, logm), func_adaf[k0:k1+1])))
    logLnu[np.isnan(logLnu)] = -100   
    if (logmdot>=logmdot_trunc) & truncation:
        logLnu_trunc = np.array(list(map(lambda f: f(logmdot, logm), func_trunc[k0:k1+1])))
        logLnu_trunc[np.isnan(logLnu_trunc)] = -100   
        logLnu = np.log10( 10**logLnu+10**logLnu_trunc )
    return logLnu.reshape(-1)


def SED_diskcor(logm, logmdot, k0=0, k1=len(lognu)-1):
    """calculate the spectral energy distribution (SED) of a disk-corona system"""
    logLnu = np.array(list(map(lambda f: f(logmdot, logm), func_diskcor[k0:k1+1])))
    logLnu[np.isnan(logLnu)] = -100   
    return logLnu.reshape(-1)



