#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:36:21 2017

@author: benjaminpujol
"""
import nltk
import preprocessing as PP
import pandas as pd
def main():
    
    #NLP settings
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    
    #Constructs child and parent graphs
    PP.improve_info()
    
    #Loading full data
    X_train = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/training_set.txt', sep = ' ', header  = None).values
    Y_train = X_train[:,2]
    X_test = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/testing_set.txt', sep = ' ', header  = None).values
    Inf = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/node_information.csv', sep = ',', header  = None).set_index(0) 
    Index = Inf.index
    
    TDIDF_title, TDIDF_abstract = PP.buildTDIDF()
    Index = pd.DataFrame(range(len(Index))).set_index(Index)