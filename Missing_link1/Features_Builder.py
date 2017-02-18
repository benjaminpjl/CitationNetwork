import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk 
from nltk.stem.porter import *
import csv
import os.path
from scipy import sparse, io

# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes


class similarity_measure:
    def __init__(self, name, stpwds):
        self.name   = name
        self.stpwds = stpwds
        self.stemmer= PorterStemmer()

    def give_score(self,obj_1, obj_2):
        obj_1,obj_2=self.process(obj_1,obj_2)
        return self.score(obj_1,obj_2)

    def process(self,obj_1,obj_2):
        obj_1 = [token for token in obj_1 if token not in self.stpwds]
        obj_1 = [self.stemmer.stem(token) for token in obj_1] 
        obj_2 = [token for token in obj_2 if token not in self.stpwds]
        obj_2 = [self.stemmer.stem(token) for token in obj_2]        
        return obj_1, obj_2

    def score(self, obj_1, obj_2):
        pass
        
class matching_sim(similarity_measure):
    def __init__(self):
        similarity_measure.__init__(self,'Matching Similarity',set(nltk.corpus.stopwords.words("english")))

    def score(self, obj_1, obj_2):
        return len(set(obj_1).intersection(set(obj_2)))

class dice_sim(similarity_measure):
    def __init__(self):
        similarity_measure.__init__(self,'Dice Similarity',set(nltk.corpus.stopwords.words("english")))

    def score(self, obj_1, obj_2):
        if (len(obj_1)+len(obj_2))== 0:
            return 0
        else :
            return 2*len(set(obj_1).intersection(set(obj_2)))/(len(obj_1)+len(obj_2))
        
class jaccard_sim(similarity_measure):
    def __init__(self):
        similarity_measure.__init__(self,'Jaccard Similarity',set(nltk.corpus.stopwords.words("english")))

    def score(self, obj_1, obj_2):
        if len(set(obj_1).union(set(obj_2))) == 0:
            return 0
        else :
            return len(set(obj_1).intersection(set(obj_2)))/len(set(obj_1).union(set(obj_2)))        

class overlap_sim(similarity_measure):
    def __init__(self):
        similarity_measure.__init__(self,'Overlap Similarity',set(nltk.corpus.stopwords.words("english")))

    def score(self, obj_1, obj_2):
        if min(len(obj_1),len(obj_2)) == 0:
            return 0
        else :
            return len(set(obj_1).intersection(set(obj_2)))/min(len(obj_1),len(obj_2))                          


class cosine_sim(similarity_measure):
    def __init__(self):
        similarity_measure.__init__(self,'Cosine Similarity',set(nltk.corpus.stopwords.words("english")))

    def score(self, obj_1, obj_2):
        if len(obj_1)*len(obj_2) == 0:
            return 0
        else :
            return len(set(obj_1).intersection(set(obj_2)))/len(obj_1)*len(obj_2)  
            
class parent_sim(similarity_measure):
    def __init__(self):
        similarity_measure.__init__(self,'Parent Similarity',set(nltk.corpus.stopwords.words("english")))

    def score(self, obj_1, obj_2):
        if len(obj_1)*len(obj_2) == 0:
            return 0
        else :
            return len(set(obj_1).intersection(set(obj_2)))/len(obj_1)*len(obj_2)         

class features:
    
    def __init__(self, info, similarity_measure, TDITF_title, TDITF_abstract, Index):
        self.info           = info
        self.sim            = similarity_measure
        self.TDITF_title    = TDITF_title
        self.TDITF_abstract = TDITF_abstract
        self.Index          = Index

    def set_tdidf(self, index):
        Inf                 = self.info.loc[index]
        vec                 = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        self.TDITF_title    = vec.fit_transform(Inf.values[:,1])
        self.TDITF_abstract = vec.fit_transform(Inf.values[:,4])
        self.Index          = pd.DataFrame(range(len(Inf))).set_index(Inf.index)
        
    def graph_child(self,Xtr):
        link_text={}
        for i in xrange(len(Xtr)):
          if Xtr[i,2]== 1 :
             if Xtr[i,0] in link_text :
               link_text["Xtr[i,0]"]=link_text["Xtr[i,0]"].append(Xtr[i,1])
             else :
               link_text["Xtr[i,0]"]=[Xtr[i,1]]   
        return link_text 
        
    def graph_parent(self,Xtr):
        link_text={}
        for i in xrange(len(Xtr)):
            if Xtr[i,2]== 1 :
               if Xtr[i,1] in link_text :
                 link_text["Xtr[i,1]"]=link_text["Xtr[i,1]"].append(Xtr[i,0])
               else :
                 link_text["Xtr[i,1]"]=[Xtr[i,0]]
            
        return link_text   

    def avg_title_tdidf(self,doc_1, doc_2):
        avg = [];
        for doc in doc_1[0:4]:
            for doc_ in doc_2[0:4]:
                avg.append(self.TDITF_title[self.Index.loc[int(doc)].values[0]].dot(self.TDITF_title[self.Index.loc[int(doc_)].values[0]].T).todense()[0,0])
        return np.mean(avg)

    def avg_abstract_tdidf(self,doc_1, doc_2):
        avg = [];
        for doc in doc_1[0:4]:
            for doc_ in doc_2[0:4]:
                avg.append(self.TDITF_abstract[self.Index.loc[int(doc)].values[0]].dot(self.TDITF_abstract[self.Index.loc[int(doc_)].values[0]].T).todense()[0,0])
        return np.mean(avg)     
         
    def gen_features(self, edges, index=False):

        if index:
            self.set_tdidf(index)

        self.edges = edges
        # Features
        temp_diff        = []
        comm_auth        = []  
        same_journal     = []  

        overlap_title    = []
        overlap_abstract = []

        abstract_tdidf   = []
        title_tdidf      = []

        same_child       = []
        same_parent      = []

        avg_child_title_tdidf   = []
        avg_parents_title_tdidf = []

        avg_child_abstract_tdidf   = []
        avg_parents_abstract_tdidf = []

        nb_source_childs = []
        nb_target_childs  = []

        nb_source_parents = []
        nb_target_parents = []


        # Looping over edges
        for i in xrange(len(self.edges)):

            source = self.edges[i][0]
            target = self.edges[i][1]

            source_info  = self.info.loc[source].fillna(str(source))
            target_info  = self.info.loc[target].fillna(str(target)) 

            # Solving technical bug for author

            #source_info[[3,4]]  = source_info[[3,4]].fillna(str(source) + 'dis')
            #target_info[[3,4]]  = source_info[[3,4]].fillna(str(target) + 'dis')     

            # Processing title
            source_title = source_info[2].lower().split(" ")        
            target_title = target_info[2].lower().split(" ")
            title_tdidf_  = self.TDITF_title[self.Index.loc[source].values[0]].dot(self.TDITF_title[self.Index.loc[target].values[0]].T).todense()[0,0]

             # Processing abstract
            source_abstract = source_info[5].lower().split(" ")
            target_abstract = target_info[5].lower().split(" ")     
            abstract_tdidf_  = self.TDITF_abstract[self.Index.loc[source].values[0]].dot(self.TDITF_abstract[self.Index.loc[target].values[0]].T).todense()[0,0]  

            # Processing author
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

            # Appending
            temp_diff.append(int(source_info[1]) - int(target_info[1]))

            overlap_title.append(self.sim.give_score(source_title,target_title))
            title_tdidf.append(title_tdidf_)
            
            overlap_abstract.append(self.sim.give_score(source_abstract,target_abstract))
            abstract_tdidf.append(abstract_tdidf_)

            same_child.append(self.sim.give_score(source_childs,target_childs))
            same_parent.append(self.sim.give_score(source_parents,target_parents))

            nb_source_childs.append(len(set(source_childs)))
            nb_target_childs.append(len(set(target_childs))) 

            nb_source_parents.append(len(set(source_parents)))
            nb_target_parents.append(len(set(target_parents)))         
            
            comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
            same_journal.append(self.sim.give_score(source_journal,target_journal))

            #avg_child_title_tdidf.append(self.avg_title_tdidf(source_childs,target_childs))
            #avg_parents_title_tdidf.append(self.avg_title_tdidf(source_parents,target_parents))

            #avg_child_abstract_tdidf.append(self.avg_abstract_tdidf(source_childs,target_childs))
            #avg_parents_abstract_tdidf.append(self.avg_abstract_tdidf(source_parents,target_parents))

                
            if (i + 1) % int(len(self.edges)/4) == 0:
                print(str(i) + "/" + str(len(self.edges)) + " samples processsed (" + str(100*i/len(self.edges)) + "%)")

        # Return final feature representation
        return np.array([overlap_title, title_tdidf, temp_diff,comm_auth,overlap_abstract, abstract_tdidf, same_journal,same_child,same_parent, \
                        nb_source_childs, nb_target_childs, nb_source_parents, nb_target_parents], dtype = np.float64).T

        #return np.array([title_tdidf, temp_diff, comm_auth, abstract_tdidf, same_journal, same_child, \
        #                same_parent, nb_source_childs, nb_target_childs, nb_source_parents, \
        #                nb_target_parents, avg_child_title_tdidf, avg_parents_title_tdidf, avg_child_abstract_tdidf, \
        #                avg_parents_abstract_tdidf ], dtype = np.float64).T

def buildTDITF():

    # File paths
    tditf_title     = 'Features/tditf_title.mtx'
    tditf_abstract  = 'Features/tditf_abstract.mtx'

    # Building term vectore representation of title and abstract
    if os.path.isfile(tditf_title) and os.path.isfile(tditf_abstract):
        return io.mmread(tditf_title).tocsr(), io.mmread(tditf_abstract).tocsr()

    elif os.path.isfile(tditf_title):

        # Generating term vectorization for abstract
        Inf   = pd.read_csv('Data/node_information.csv', header=None).set_index(0).values
        vec   = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        Inf   = vec.fit_transform(Inf[:,4])
        io.mmwrite(tditf_abstract, Inf)

        return io.mmread(tditf_title).tocsr(), Inf.tocsr()

    elif os.path.isfile(tditf_abstract):

        # Generating term vectorization for title
        Inf   = pd.read_csv('Data/node_information.csv', header=None).set_index(0).values
        vec   = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        Inf   = vec.fit_transform(Inf[:,1])
        io.mmwrite(tditf_title, Inf)

        return Inf.tocsr(), io.mmread(tditf_abstract).tocsr()
    else :

        # Generating term vectorization for both
        Inf     = pd.read_csv('Data/node_information.csv', header=None).set_index(0).values
        vec     = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        # Title
        Inf_title     = vec.fit_transform(Inf[:,1])
        io.mmwrite(tditf_title, Inf_title)      
        # Abstract
        Inf_abstract   = vec.fit_transform(Inf[:,4])
        io.mmwrite(tditf_abstract, Inf_abstract)    

        return Inf_title.tocsr(), Inf_abstract.tocsr()


def improve_info():

    # Loading node info and training set
    Xtr   = pd.read_csv('Data/training_set.txt', sep=' ', header=None).values
    Inf   = pd.read_csv('Data/node_information.csv', header=None).values
    if Inf.shape[1] == 8:
        print('Childs and Parents graphs already set in node_information.csv')
    else :
        print('Starting generation of parents and childs graphs ')
        Index = pd.DataFrame(range(len(Inf[:,0]))).set_index(Inf[:,0])

        # Preparing parents and child 1-D arrays
        parents = np.chararray((len(Inf),1), itemsize=10000)
        childs  = np.chararray((len(Inf),1), itemsize=10000)
        i = 0
        # Looping over Xtr
        for edge in Xtr :
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
            if (i + 1) % int(len(Xtr)/4) == 0:
                print(str(i) + "/" + str(len(Xtr)) + " samples processsed (" + str(100*i/len(Xtr)) + "%)")
            i = i + 1

        # Improving node info file
        Inf = np.hstack([Inf, parents, childs])
        Inf = pd.DataFrame(Inf)
        Inf.to_csv('Data/node_information.csv' , header=False, index=False)
        print('Childs and parents graph have been generated and stored on disk')














