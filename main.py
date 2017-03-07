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
from sklearn import svm
from sklearn import preprocessing
import numpy as np

_k = 0.001
SINGLE_RUN = 1

def GetPrediction(X_train, Y_train, X_test, classifier = None):

        #Initializing default linear SVM Classifier
        if classifier == None:
            classifier = svm.LinearSVC()

        #Preprocessing data: Center to the mean and component wise scale to unit variance
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)

        #Training classifier
        print("Fitting started")
        classifier.fit(X_train, Y_train)

        print("Fitting over")
        print("Starting prediction")
        
        print classifier.score(X_train, Y_train)

        return classifier.predict(X_test)

def WriteSubmission(Pred, loc):
        df = pd.DataFrame(Pred)
        df.columns = ['category']
        df.index.name = 'id'
        df.to_csv(loc)





def main():
    
    
#    
#    #Constructs child and parent graphs
#    #FB.improve_info()
#    
#    #Loading full data 
    X_train = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/training_set.txt', sep = ' ', header  = None)
    Y_train = X_train[:,2]
#    X_test = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/testing_set.txt', sep = ' ', header  = None).values
#    Inf = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/node_information.csv', sep = ',', header  = None).set_index(0) 
#    Index = Inf.index
#    
#    TDIDF_title, TDIDF_abstract = FB.buildTDIDF()
#    Index = pd.DataFrame(range(len(Index))).set_index(Index)
#    
#    
#    if SINGLE_RUN: 
#        #Selecting only subsets of the data
#        to_keep = random.sample(range(len(X_train)), k = int(round(len(X_train)*_k)))
#        X_train = X_train[to_keep]
#        Y_train = Y_train[to_keep] 
#        
#        index_train = set(X_train[:,0]).union(set(X_train[:,1]))
#        index_test =  set(X_test[:,0]).union(set(X_test[:,1]))
#        
#    
#   
#    #Initiating similarity and feature_builder class
#    
#    Similarity = FB.matching_sim()
#    Feature_Builder = FB.features(Inf, Similarity, TDIDF_title, TDIDF_abstract, Index)
    
    #Computing features
    print('-----------------------------------------------------')
    print('Computing features for X_train')
    #X_train_features = Feature_Builder.gen_features(X_train, index_train)
    X_train_features = np.load("/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/X_train_features.npy")
    X_train_features= X_train_features[:6000]
    Y_train[:6000]
    
    print('-----------------------------------------------------')
    print("Computing features for  X_test")
    #X_test_features  = Feature_Builder.gen_features(X_test, index_test)
    X_test_features = np.load("/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/X_test_features.npy")

    #Getting prediction for linear SVM

    Classifier = svm.SVC(kernel ='rbf', gamma = 2)

    Pred = GetPrediction(X_train_features, Y_train, X_test_features, Classifier)
    
    

    #Comparing to best to date submission

    ##Fill the code here##

    #Write submission file
    #WriteSubmission(Pred, '/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/submission.csv')


  



if __name__ == "__main__":
    main()

