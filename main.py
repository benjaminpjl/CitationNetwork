#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:36:21 2017

@author: benjaminpujol
"""
import nltk
import preprocessing as PP
import pandas as pd
import random
import networkx as nx


_k = 0.00001
SINGLE_RUN = 1


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
    
    #TDIDF_title, TDIDF_abstract = PP.buildTDIDF()
    Index = pd.DataFrame(range(len(Index))).set_index(Index)
    
    
    if SINGLE_RUN: 
        #Selecting only subsets of the data
        to_keep = random.sample(range(len(X_train)), k = int(round(len(X_train)*_k)))
        X_train = X_train[to_keep]
        Y_train = Y_train[to_keep]
        print X_train
        
        index_train = set(X_train[:,0]).union(set(X_train[:,1]))
        index_test =  set(X_test[:,0]).union(set(X_test[:,1]))
        
        return index_train, index_test

    index_train, index_test = main()
    print index_test
    
    #Creating a graph
    G = nx.graph()
    #Adding nodes and edges
    ind = X_train[:,2]==1
    G.add_nodes_from(Inf.index)  #Add every documents as a node
    G.add_edges_from(X_train[ind][:,[0,1]]) #Add edges 
    pagerank = nx.pagerank(G, alpha = 0.8) #PageRank computes a ranking of the nodes in the graph G based on the structure of incoming links
    
    #Initiating similarity and feature class
    
    similarity = PP.matching_sim()


