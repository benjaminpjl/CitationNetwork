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
    X_train = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/training_set.txt', sep = ' ', header  = None).values
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
        
        #Looping over X_rain
        for edge in X_train:
            if edge[2] == 1:
                #Updating parents array
                if parents[Index.loc[edge[1]][0]][0] == '':
                    parents[Index.loc[edge[1]][0]] == str(edge[0])
                else:
                    parents[Index.loc[edge[1]][0]] == parents[Index.loc[edge[1]][0]] + " " + str(edge[0])
                
                    
                # Updating childs array
                if childs[Index.loc[edge[0]][0]][0]   == '':
                    childs[Index.loc[edge[0]][0]]   = str(edge[1])
                else :
                    childs[Index.loc[edge[0]][0]]   = childs[Index.loc[edge[0]][0]]   + " " + str(edge[1])

            
            if (i + 1) % int(len(X_train)/10) == 0:
                print(str(i) + "/" + str(len(X_train)) + " samples processsed (" + str(100*i/len(X_train)) + "%)")
            
            i = i+1
            
        #Improving node file info
        Inf = np.hstack([Inf, parents, childs])
        Inf = pd.DataFrame(Inf)
        Inf.to_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/node_information.csv', header=False, index=False)
        print('Childs and parents graph have been generated and stored on disk')

improve_info()
            