#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:13:40 2017

@author: benjaminpujol
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
from scipy import io, sparse
import csv
import nltk
from nltk.stem.porter import PorterStemmer

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
         
        for edge in X_train :
            if edge[2] == 1:
                # Updating parents array

                if parents[Index.loc[edge[1]][0]][0] == '':
                    parents[Index.loc[edge[1]][0]] = str(edge[0])
                else :
                    parents[Index.loc[edge[1]][0]] = parents[Index.loc[edge[1]][0]] + " " + str(edge[0])

                # Updating childs array
                if childs[Index.loc[edge[0]][0]][0]   == '':
                    childs[Index.loc[edge[0]][0]]   = str(edge[1])
                else :
                    childs[Index.loc[edge[0]][0]]   = childs[Index.loc[edge[0]][0]]   + " " + str(edge[1])
                    
            if (i + 1) % int(len(X_train)/4) == 0:
                print(str(i) + "/" + str(len(X_train)) + " samples processsed (" + str(100*i/len(X_train)) + "%)")
            i = i + 1
            
        print parents.shape
        print childs.shape
        print Inf.shape
            
        #Improving node file info
        Inf = np.hstack((Inf, parents, childs))  #PROBLEM HERE!!!
        Inf = pd.DataFrame(Inf)
        Inf.to_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/node_information.csv', header=False, index=False)
        print('Childs and parents graph have been generated and stored on disk')

improve_info()

def buildTDIDF():
   
    # File paths
    tdidf_title = '/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/tdidf_title.mtx'
    tdidf_abstract = '/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/tdidf_abstract.mtx'
    
    # Building term vector representation of title and abstract
    if os.path.isfile(tdidf_title) and os.path.isfile(tdidf_abstract):
        return io.mmread(tdidf_title).tocsr(), io.mmread(tdidf_abstract).tocsr()
    
    elif os.path.isfile(tdidf_title):
        #Generating term vectorisation for abstract
        Inf = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/node_information.csv', header = None).set_index(0).values
        vec = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = "english" )  #Apply sublinear tf scaling ie replace tf -> 1 + log(tf) and ignore words that appear in more than 50% of docs
        Inf = vec.fit_transform(Inf[:,2]) #Return term-document matrix
        io.mmwrite(tdidf_abstract, Inf)
        
        return io.mmread(tdidf_title).tocsr(), Inf.tocsr() #Return matrix in a compressed sparse row format (csr)
    
    elif os.path.isfile(tdidf_abstract):

        # Generating term vectorization for title
        Inf = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/node_information.csv', header = None).set_index(0).values
        vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        Inf = vec.fit_transform(Inf[:,1])
        io.mmwrite(tdidf_title, Inf)

        return Inf.tocsr(), io.mmread(tdidf_abstract).tocsr()
        
    else:
        
        # Generating term vectorization for both
        Inf = pd.read_csv('/Users/benjaminpujol/Documents/Cours3A/CitationNetwork/node_information.csv', header = None).set_index(0).values
        vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        # Title
        Inf_title = vec.fit_transform(Inf[:,1])
        io.mmwrite(tdidf_title, Inf_title)      
        # Abstract
        Inf_abstract   = vec.fit_transform(Inf[:,4])
        io.mmwrite(tdidf_abstract, Inf_abstract)    

        return Inf_title.tocsr(), Inf_abstract.tocsr()
        
class similarity_measure:
    def __init__(self, name, stpwds):
        self.name = name
        self.stpwds = stpwds
        self.stemmer = PorterStemmer()
    
    def give_score(self, obj1, obj2):
        obj1, obj2 = self.process(obj1, obj2)
        return self.score(obj1, obj2)
        
    def process(self, obj1, obj2):
        obj1 = [token for token in obj1 if token not in self.stpwds]
        obj1 = [self.stemmer.stem(token) for token in obj1]
        obj2 = [token for token in obj2 if token not in self.stpwds]
        obj2 = [self.stemmer.stem(token) for token in obj2]
        return obj1, obj2
    
    def score(self, obj1, obj2):
        pass

class matching_sim(similarity_measure):
    def __init__(self):
        similarity_measure.__init__(self, 'Matching Similarity', set(nltk.corpus.stopwords("english")))
   
    def score(self,obj1,obj2):
        return len(set(obj1).intersection(set(obj2)))
        
        
        
##INSERT OTHER SIMILARITY CLASSES HERE##
        
        
class features:
    def __init__(self, info, similarity_measure, TDIDF_title, TDIDF_abstract, Index)
        self.info = info
        self.similarity_measure = similarity_measure
        self.TDIDF_title = TDIDF_title
        self.TDIDF_abstract = TDIDF_abstract
        self.Index = Index
    
    def set_tdidf(self,index):
        Inf = self.info.loc[index]
        vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english")
        self.TDIDF_title = vec.fit_transform(Inf.values[:,1])
        self.TDIDF_abstract = vec.fit_transform(Inf.values[:,4])     
        self.Index = pd.DataFrame(range(len(Inf))).set_index(Inf.index)
        
    def graph_child(self, X_train):
        link_text = {}
        for i in xrange(len(X_train)):
            if X_train[i,2]==1 :
                if X_train[i,0] in link_text:
                    link_text["X_train[i,0]"] = link_text["X_train[i,0]"].append(X_train[i,1])
                else:
                    link_text["X_train[i,O]"] = [X_train[i,1]]
        return link_text 
            
    
    def graph_parent(self, X_train):
        link_text = {}
        for i in xrange(len(X_train)):
            if X_train[i,2]==1 :
                if X_train[i,1] in link_text:
                    link_text["X_train[i,1]"] = link_text["X_train[i,1]"].append(X_train[i,0])
                else:
                    link_text["X_train[i,1]"] = [X_train[i;0]]
        return link_text
        
        
        
    
        
        
    def gen_features(self, edges, index = False):
        
        if index:
            self.set_tdidf(index)
        
        self.edges = edge
        
        #Features
        
        temp_diff = []  #Time difference
        comm_author = []
        same_journal = []

        overlap_title = []
        overlap_abstract = []

        title_tfidf = []
        abstract_tdidf = []

        same_child = []
        same_parent = []

        nb_source_childs = []
        nb_source_parents = []

        nb_target_childs = []
        nb_target_parents = []

        #Looping over edges
        for i in xrange (len(self.edges)):
            
            source = self.edges[i][0]
            target = self.edges[i][1]

            source_info = self.info.loc[source].fillna(str(source))
            target_info = self.info.loc[target].fillna(str(target))
            
            #Processing titles
            source_title = source_info[2].lower.split(" ")
            target_title = target_info[2].lower.split(" ")
            title_tdidf_  = self.TDITF_title[self.Index.loc[source].values[0]].dot(self.TDITF_title[self.Index.loc[target].values[0]].T).todense()[0,0]
            

            #Processing abstracts
            
            source_abstract = source_info[5].lower.split(" ")
            target_abstract = target_info[5].lower.split(" ")
            abstract_tdidf_  = self.TDITF_abstract[self.Index.loc[source].values[0]].dot(self.TDITF_abstract[self.Index.loc[target].values[0]].T).todense()[0,0]
            
            #Processing authors
            source_auth = source_info[3].split(",")
            target_auth = target_info[3].split(",")
            
            # Processing journal of publication
            source_journal = source_info[4].split(".")
            target_journal = target_info[4].split(".")
            
            # Processing children and parents
            source_parents = source_info[6].split(" ")
            target_parents = target_info[6].split(" ")  
            
            source_childs  = source_info[7].split(" ")
            target_childs  = target_info[7].split(" ") 
            
            # Appending to features
            
            temp_diff.append(int(source_info[1])-int(target_info[1]))
            
            overlap_title.append(self.sim.give_score(source_title, target_title))
            title_tfidf.append(title_tdidf_)
            
            overlap_abstract.append(self.sim.give_score(source_abstract, target_abstract))
            abstract_tdidf.append(abstract_tdidf_)
            
            same_child.append(self.sim.give_score(source_childs, target_childs))
            same_parent.append(self.sim.give_score(source_parents, target_parents))
            
            nb_source_childs.append(len(set(source_childs)))
            nb_target_childs.append(len(set(target_childs))) 
    
            nb_source_parents.append(len(set(source_parents)))
            nb_target_parents.append(len(set(target_parents))) 
            
            comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
            same_journal.append(self.sim.give_score(source_journal,target_journal))
            
            if (i + 1) % int(len(self.edges)/4) == 0:
                    print(str(i) + "/" + str(len(self.edges)) + " samples processsed (" + str(100*i/len(self.edges)) + "%)")
            
    # Return final feature representation
        return np.array([overlap_title, title_tdidf, temp_diff,comm_auth,overlap_abstract, abstract_tdidf, same_journal,same_child,same_parent, \ 
                         nb_source_childs, nb_target_childs, nb_source_parents, nb_target_parents], dtype = np.float64).T
            