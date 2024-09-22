import numpy as np
from scipy.optimize import fsolve

def bin_trace(data, bin_edges):
    """identifies the indices of data points that fall into each bin"""
    if bin_edges[-1]<np.max(data): raise ValueError('the last bin_edges should be greater than maximum value of data')
    bin_indices = np.digitize(data, bin_edges)
    return [ np.argwhere(bin_indices == i + 1).reshape(-1)
        for i in range(len(bin_edges) - 1) ]

def linear_interp(x, x1, x2, y1, y2):
    return (y2-y1)/(x2-x1) * (x-x1) + y1

# bolometric corrections in Marconi+2004
def func_HX(x, logL_HX):
    return -1.54-0.24*x-0.012*x**2+0.0015*x**3 - logL_HX + x + 12.+np.log10(3.83e33)

def func_SX(x, logL_SX):
    return -1.65-0.22*x-0.012*x**2+0.0015*x**3 - logL_SX + x + 12.+np.log10(3.83e33)

def convert2bol(logL, func):
    logL_bol = []
    for i in range(len(logL)):
        logL_bol.append( fsolve(func, [1.], args=logL[i])[0] + 12+np.log10(3.83e33) )
    return np.array(logL_bol)