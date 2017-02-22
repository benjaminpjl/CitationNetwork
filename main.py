#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:36:21 2017

@author: benjaminpujol
"""
import nltk
import pandas as pd
import random
import networkx as nx
import sys
sys.path.append('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/')
import features_builder as FB


_k = 0.001
SINGLE_RUN = 1


def main():
    
    
    
    #Constructs child and parent graphs
    #FB.improve_info()
    
    #Loading full data
    X_train = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/training_set.txt', sep = ' ', header  = None).values
    Y_train = X_train[:,2]
    X_test = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/testing_set.txt', sep = ' ', header  = None).values
    Inf = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/node_information.csv', sep = ',', header  = None).set_index(0) 
    Index = Inf.index
    
    TDIDF_title, TDIDF_abstract = FB.buildTDIDF()
    Index = pd.DataFrame(range(len(Index))).set_index(Index)
    
    
    if SINGLE_RUN: 
        #Selecting only subsets of the data
        to_keep = random.sample(range(len(X_train)), k = int(round(len(X_train)*_k)))
        X_train = X_train[to_keep]
        Y_train = Y_train[to_keep]
        print X_train
        
        index_train = set(X_train[:,0]).union(set(X_train[:,1]))
        index_test =  set(X_test[:,0]).union(set(X_test[:,1]))
        
    
    #Creating a graph
    G = nx.Graph()
    #Adding nodes and edges
    ind = X_train[:,2]==1
    G.add_nodes_from(Inf.index)  #Add every documents as a node
    G.add_edges_from(X_train[ind][:,[0,1]]) #Add edges 
    pagerank = nx.pagerank(G, alpha = 0.8) #PageRank computes a ranking of the nodes in the graph G based on the structure of incoming links
    
    #Initiating similarity and feature class
    
    Similarity = FB.matching_sim()
    Feature_Builder = FB.features(Inf, Similarity, TDIDF_title, TDIDF_abstract, Index)
    
    #Computing features
    print('-----------------------------------------------------')
    print('Computing features for X_train')
    X_train_features = Feature_Builder.gen_features(X_train, index_train)
    print X_train_features[:10,:]


main()

