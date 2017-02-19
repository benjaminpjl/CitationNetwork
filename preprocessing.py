#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:13:40 2017

@author: benjaminpujol
"""

import pandas as pd
import numpy as np

def improve_info():
    
    #Loading node info and training set
    Xtr = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/training_set.txt', sep = ' ', header  = None).values
    Inf = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/node_information.csv', sep = ',', header  = None).values
    
    if Inf.shape[1] == 7:
        print('Childs and Parents graphs already set in node_information.csv')
    else :
        print('Starting generation of parents and childs graphs ')
        Index = pd.DataFrame(range(len(Inf[:,0]))).set_index(Inf[:,0])
        
        #preparing parents and childs 1D arrays
        parents = np.chararray((len(Inf), 1), itemsize = 10000)
        childs = np.chararray((len(Inf), 1), itemsize = 10000)
        i=0
