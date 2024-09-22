import numpy as np
import pickle
import importlib.resources

lognu = np.round(np.arange(8., 23., 0.1), 1)
logmdot_crit = -2.9
logmdot_trunc= -3.0 


func_adaf = []
with importlib.resources.open_binary('agnSED.data', 'save_adaf.pck') as file:  
    for i in range(len(lognu)):
        func_adaf.append(pickle.load(file))
        
func_trunc = []
with importlib.resources.open_binary('agnSED.data', 'save_trunc.pck') as file:
    for i in range(len(lognu)):
        func_trunc.append(pickle.load(file))
        
func_diskcor = []
with importlib.resources.open_binary('agnSED.data', 'save_diskcor.pck') as file:
    for i in range(len(lognu)):
        func_diskcor.append(pickle.load(file))
        
func_slim = []
with importlib.resources.open_binary('agnSED.data', 'save_slim.pck') as file:
    for i in range(len(lognu)):
        func_slim.append(pickle.load(file))
        
func_SSD = []
with importlib.resources.open_binary('agnSED.data', 'save_wholeSSD.pck') as file:    
    for i in range(len(lognu)):
        func_SSD.append(pickle.load(file))