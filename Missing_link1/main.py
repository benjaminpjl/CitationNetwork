import random
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn import svm, grid_search
from sklearn.metrics.pairwise import linear_kernel
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
import sklearn.ensemble as se
import nltk 
from nltk.stem.porter import *
import csv
import Features_Builder as FB
import os

import xgboost as xgb
import networkx as nx


_k          = 100e-2
sz_te       = 32648 # 32648
SINGLE_RUN  = 1
X_VAL_SNG   = 0
LEARN_SIM   = 0
X_VAL_CLA   = 0
SZ_IMPACT   = 0


def main():

    # NLP settings
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))

    # Constructs childs and parents graph
    FB.improve_info()

    # Loading full data
    #X_train=pd.read_csv('Data/training_set.txt', sep=' ', header=None).values
    Xtr = pd.read_csv('Data/training_set.txt', sep=' ', header=None).values
    Ytr = Xtr[:,2]
    Xte = pd.read_csv('Data/testing_set.txt' , sep=' ', header=None).values
    Inf = pd.read_csv('Data/node_information.csv', header=None).set_index(0)
    Index = Inf.index

    TDITF_title, TDITF_abstract = FB.buildTDITF()
    Index = pd.DataFrame(range(len(Index))).set_index(Index)

    if SINGLE_RUN :
        # Selecting subsets
        #To_keep = random.sample(range(len(Xtr)), k=int(round(len(Xtr)*_k)))
        #Xtr     = Xtr[To_keep]
        #Ytr     = Ytr[To_keep]
        #Xte     = Xte[:sz_te ]

        index_tr = set(Xtr[:,0]).union(set(Xtr[:,1]))
        index_te = set(Xte[:,0]).union(set(Xte[:,1]))

        # Construct graph
        G   =  nx.Graph()
        # Adding nodes et edges
        ind = Xtr[:,2]==1
        G.add_nodes_from(Inf.index)
        G.add_edges_from(Xtr[ind][:,[0,1]])
        pagerank = nx.pagerank(G, alpha=0.8)


        # Initiating Similarity and Feature class

        columns  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

        Similarity      = FB.matching_sim()
        Feature_Builder = FB.features(Inf, Similarity, TDITF_title, TDITF_abstract, Index)
        print('----------------------------------------------------------')
        print('Using ' + Similarity.name)

        # Computing features 
        print('----------------------------------------------------------')
        print('Computing features for Xtr')
        #Xtr_features = Feature_Builder.gen_features(Xtr, index_tr)

        Xtr_features = np.load('Features/All_feats_Xtr.npy')#,X_train)
        #Xtr_features = Xtr_features[:,columns]
        
        #Xtr_features = Xtr_features[To_keep]
        #np.save('Features/All_feats_Xtr.npy', Xtr_features) !!!!!! A utiliser si tu veux saver les features et ne plus les recalculer a chaque fois 
        print('----------------------------------------------------------')
        print('Computing features for Xte')
        #Xte_features = Feature_Builder.gen_features(Xte, index_te)

        Xte_features = np.load('Features/All_feats_Xte.npy')#,X_train)
        #Xte_features = Xte_features[:,columns]
        #Xte_features = Xte_features[:sz_te]
        #np.save('Features/All_feats_Xte.npy', Xte_features)
        print('----------------------------------------------------------')

        # Getting prediction for Linear SVM
        
        #Classifier = svm.SVC(kernel='rbf', gamma=2)
        #dict_random={"n_estimators":[10,20,30,50,100,125,200], "criterion":["gini","entropy"], "max_depth":[10,50,100,150,200], "min_samples_split":[2,5],  \
        #               "bootstrap":[True,False]}
        #Classifier =RandomizedSearchCV(se.RandomForestClassifier(), param_distributions=dict_random, n_iter=30, scoring="f1", \
        #                   refit=True, cv=None, verbose=3, random_state=None)
        #Classifier = se.RandomForestClassifier(n_estimators=100)
        #Classifier = se.GradientBoostingClassifier(n_estimators=100)#, learning_rate=1.0, max_depth=1, random_state=0)

        Pred       = GetPrediction(Xtr_features, Ytr, Xte_features, Classifier)
        #Pred       = GetPrediction_(Xtr_features, Ytr, Xte_features)

        # Comparing to best to date submission
        best_to_date    = pd.read_csv('Prediction/submission_best_.csv').values[:,1]
        Acc             = best_to_date == Pred
        print('Relative accuracy with best to date prediction : ' + str(int(100*np.mean(Acc))) + '%')

        # Writing submission file
        WriteSubmission(Pred,'Prediction/submission.csv')

    if SZ_IMPACT :

        # List of size
        sizes = np.array([0.5,1])*1e-2#,15,20])*1e-2#,25,30,35,40,45,50])*1e-2

        # Setting feature class
        Feature_Builder = FB.features(Inf, FB.matching_sim(), TDITF_title, TDITF_abstract, Index)

        # List of classifiers to test
        classifiers  = [se.AdaBoostClassifier(n_estimators=100), se.RandomForestClassifier(n_estimators=100),se.RandomForestClassifier(n_estimators=150)]
        class_ids    = ['Ada Boost N_tree 100', 'Rnd Forest N_tree 100', 'Rnd Forest N_tree 150']

        # All features
        All_features = np.load('Features/All_feats_Xtr.npy')

        # Fixed validation set and labels
        Xval_        = Xtr[-sz_te:]
        Yval_        = Ytr[-sz_te:]
        index_val    = set(Xval_[:,0]).union(set(Xval_[:,1]))

        # Generating features for testing set
        print('----------------------------------------------------------')
        print('Generating features for validation set')  
        Xval_feats   = Feature_Builder.gen_features(Xval_,index_val)

        # Prepping CSV results file
        Accuracy     = np.array(['Size','Model','Acc_1','Acc_2','Acc_3','Acc_4','Acc_5','Mean Accuracy'])

        # Folding training set
        #Nb_folds     = 5
        #KFold_       = KFold(len(Ytr), n_folds=Nb_folds)

        # Looping over sizes
        for k in sizes:
            print('----------------------------------------------------------')
            print('----------------------------------------------------------')
            print('Running with size '+ str(k*100) + '%')

            i = 0
            for classifier in classifiers : 
                Accuracy_ = []
                for t in xrange(5):
                    print('----------------------------------------------------------')
                    print('----------------------------------------------------------')
                    print('Random sampling number '+ str(t))

                    to_keep = random.sample(range(len(Xtr)-sz_te), k=int(round(len(Xtr)*k)))
                    Xtr_     = Xtr[to_keep]
                    Ytr_     = Ytr[to_keep]

                    index_tr = set(Xtr[:,0]).union(set(Xtr[:,1]))

                    # Computing features 
                    print('----------------------------------------------------------')
                    print('Computing features for Xtr')
                    Xtr_feats = Feature_Builder.gen_features(Xtr_, index_tr)
                
                    print('Starting prediction for ' + class_ids[i])
                    Pred = GetPrediction(Xtr_feats, Ytr_, Xval_feats, classifier)
                    Acc  = np.mean(Yval_ == Pred)
                    Accuracy_.append(Acc)

                Accuracy_.append(sum(Accuracy_)/len(Accuracy_))
                Accuracy_.insert(0,class_ids[i])
                Accuracy_.insert(0,str(k*100) +'%')
                Accuracy    = np.vstack([Accuracy,Accuracy_])
                i = i + 1

        # Writing final results
        Accuracy_df = pd.DataFrame(Accuracy)
        Accuracy_df.to_csv("X_Val/Size_of_training_impact_final.csv" , header=False, index=False)

    if X_VAL_SNG :

        # # Selecting subsets
        To_keep = range(int(round(len(Xtr)*_k)))
        Xtr     = Xtr[To_keep]
        Ytr     = Ytr[To_keep]

        index_tr = set(Xtr[:,0]).union(set(Xtr[:,1]))
        index_te = set(Xte[:,0]).union(set(Xte[:,1]))

        # List of similarity measures to be tested
        Similarities = [FB.matching_sim(),]# FB.dice_sim(), FB.jaccard_sim(), FB.overlap_sim(), FB.cosine_sim()]

        # Folding training set
        Nb_folds     = 5
        KFold_       = KFold(len(Ytr), n_folds=Nb_folds)

        # Setting linaer SVM classifier
       # Classifier   = svm.SVC(kernel='rbf', gamma=2)
       # dic_random_forest={"n_estimators":[10,20,30,60,80,100,125,150,200],"criterion":['gini','entropy'],"max_depth":[20,30,50,100,120,160,200],"bootstrap":[True,False], "random_state":['None']}
        #Classifier=se.RandomForestClassifier(n_estimators=[10,20,30,60,80,100,125,150,200],criterion=['gini','entropy'],max_depth=[20,30,50,100,120,160,200],bootstrap=[True,False], random_state=None)
        Classifier=se.RandomForestClassifier(n_estimators=100)
        Accuracy     = np.array(['Similarity','Acc_1','Acc_2','Acc_3','Acc_4','Acc_5','Mean Accuracy'])



        print('----------------------------------------------------------')
        print('----------------------------------------------------------')
        print('                  Starting X Validation')
        print('----------------------------------------------------------')
        print('----------------------------------------------------------')

        for Similarity in Similarities:

            print('                   ' + Similarity.name)

            # Setting Feature class
            Feature_Builder = FB.features(Inf, Similarity, TDITF_title, TDITF_abstract, Index)
            print ('Features generation for complete set')

            All_features    = Feature_Builder.gen_features(Xtr, index_tr)

            Accurary_    = []
            i = 1
            # X-validation
            for Ind_tr, Ind_val in KFold_:
                print('----------------------------------------------------------')
                print('               X validation number ' + str(i))
                # Splitting set
                Xtr_  = Xtr[Ind_tr ]
                Ytr_  = Ytr[Ind_tr ]
                Xval_ = Xtr[Ind_val]
                Yval_ = Ytr[Ind_val]

                # Building feature representation
                print ('Features selection for training set')
                #Xtr_features  = Feature_Builder.gen_features(Xtr_)#,Xtrain )
                Xtr_features  = All_features[Ind_tr]
                print ('Features selection for validation set')
                #Xval_features = Feature_Builder.gen_features(Xval_)#,Xtrain)
                Xval_features = All_features[Ind_val]

                # Getting predictions and accuracy score
                Pred       = GetPrediction(Xtr_features, Ytr_, Xval_features, Classifier)
                Acc        = np.mean(Yval_ == Pred)

                Accurary_.append(Acc)
                i += 1

            # Appending mean and similarity
            Accurary_.append(sum(Accurary_)/len(Accurary_))
            Accurary_.insert(0,Similarity.name)
            Accuracy    = np.vstack([Accuracy,Accurary_])
            print('----------------------------------------------------------')

        # Writing final results

        Accuracy_df = pd.DataFrame(Accuracy)
        Accuracy_df.to_csv("X_Val/Max_Graph_X_Val_RBF.csv" , header=False, index=False)

    if LEARN_SIM :

        # # Selecting subsets
        To_keep = random.sample(range(len(Xtr)), k=int(round(len(Xtr)*_k)))
        Xtr     = Xtr[To_keep]
        Ytr     = Ytr[To_keep]

        # List of similarity measures to be tested
        Similarities = [FB.dice_sim(), FB.jaccard_sim(), FB.overlap_sim(), FB.cosine_sim()]

        # Folding training set
        Nb_folds     = 5
        KFold_        = KFold(len(Ytr), n_folds=Nb_folds)

        # Setting linaer SVM classifier
        Classifier   = svm.SVC(C=1.0, kernel='rbf', gamma='0.003')
        Accuracy     = np.array(['Similarity','Acc_1','Acc_2','Acc_3','Acc_4','Acc_5','Mean Accuracy'])

        print('----------------------------------------------------------')
        print('----------------------------------------------------------')
        print('                  Starting X Validation')
        print('----------------------------------------------------------')
        print('----------------------------------------------------------')

        print('                    All similarities                      ')

        # Setting Feature class
        Feature_Builder_0 = FB.features(Inf, FB.matching_sim())

        Accurary_    = []
        i = 1
        # X-validation
        for Ind_tr, Ind_val in KFold_:
            print('----------------------------------------------------------')
            print('                   X validation number ' + str(i)          )

            # Splitting set
            Xtr_  = Xtr[Ind_tr ]
            Ytr_  = Ytr[Ind_tr ]
            Xval_ = Xtr[Ind_val]
            Yval_ = Ytr[Ind_val]

            # Building feature representation
            print ('Features generation for training set')
            Xtr_features  = Feature_Builder_0.gen_features(Xtr_,Xtrain )
            print ('Features generation for validation set')
            Xval_features = Feature_Builder_0.gen_features(Xval_,Xtrain)

            for Similarity in Similarities:
                Feature_Builder  = FB.features(Inf, Similarity)
                print ('Features generation for training set')
                Xtr_features     = np.hstack([Xtr_features ,Feature_Builder.gen_features(Xtr_,Xtrain)])
                print ('Features generation for validation set')
                Xval_features    = np.hstack([Xval_features,Feature_Builder.gen_features(Xval_,Xtrain)])

            # Getting predictions and accuracy score
            Pred       = GetPrediction(Xtr_features, Ytr_, Xval_features, Classifier)
            Acc        = np.mean(Yval_ == Pred)

            Accurary_.append(Acc)
            i += 1

        # Appending mean and similarity
        Accurary_.append(sum(Accurary_)/len(Accurary_))
        Accurary_.insert(0,'All similarities')
        Accuracy    = np.vstack([Accuracy,Accurary_])
        print('----------------------------------------------------------')

        # Writing final results

        Accuracy_df = pd.DataFrame(Accuracy)
        Accuracy_df.to_csv("X_Val/CP_Popularity_X_Val_RBF.csv" , header=False, index=False)   

    if X_VAL_CLA :

        # Selecting subsets
        To_keep = range(int(round(len(Xtr)*_k)))
        Xtr     = Xtr[To_keep]
        Ytr     = Ytr[To_keep]

        # List of classifiers to be tested
        Gamma = 3

        Similarity = FB.matching_sim()
        # Folding training set
        Nb_folds      = 5
        KFold_        = KFold(len(Ytr), n_folds=Nb_folds)

        Feature_Builder = FB.features(Inf, Similarity, TDITF_title, TDITF_abstract, Index)

        # Setting linaer SVM classifier
        Accuracy     = np.array(['Number of trees','Acc_1','Acc_2','Acc_3','Acc_4','Acc_5','Mean Accuracy'])
        #Accuracy     = []

        print('----------------------------------------------------------')
        print('----------------------------------------------------------')
        print('                  Starting X Validation')
        print('----------------------------------------------------------')
        print('----------------------------------------------------------')

        for c in range(1,Gamma):
            Classifier   = svm.SVC(kernel='rbf', gamma=c)
            print('                  Gamma =' + str(c))

            Accurary_    = []
            i = 1
            # X-validation
            for Ind_tr, Ind_val in KFold_:
                print('----------------------------------------------------------')
                print('               X validation number ' + str(i))
                # Splitting set
                Xtr_  = Xtr[Ind_tr ]
                Ytr_  = Ytr[Ind_tr ]
                Xval_ = Xtr[Ind_val]
                Yval_ = Ytr[Ind_val]

                # Building feature representation
                print ('Features generation for training set')
                Xtr_features  = Feature_Builder.gen_features(Xtr_ )
                print ('Features generation for validation set')
                Xval_features = Feature_Builder.gen_features(Xval_)

                # Getting predictions and accuracy score
                Pred       = GetPrediction(Xtr_features, Ytr_, Xval_features, Classifier)
                Acc        = np.mean(Yval_ == Pred)

                Accurary_.append(Acc)
                i += 1

            # Appending mean and similarity
            Accurary_.append(sum(Accurary_)/len(Accurary_))
            Accurary_.insert(0,'Gamma = ' + str(c))
            Accuracy    = np.vstack([Accuracy,Accurary_])
            print('----------------------------------------------------------')

        # Writing final results

        Accuracy_df = pd.DataFrame(Accuracy)
        Accuracy_df.to_csv("X_Val/Gamma_X_Val_RBF.csv" , header=False, index=False)

    # Warning for end of program
    os.system('say "your program has finished"')


def GetPrediction(Xtr, Ytr, Xte, classifier = None):

    # Initializing default linear SVM classifier
    if classifier == None :
        classifier = svm.LinearSVC()

    # Preprocessing data
    Xtr = preprocessing.scale(Xtr)
    Xte = preprocessing.scale(Xte)

    # Training classifier
    print('Fitting started')
    classifier.fit(Xtr, Ytr)

    print('Fitting Over')
    print('Starting prediction')

    return classifier.predict(Xte)


def GetPrediction_(Xtr, Ytr, Xte):


    # Preprocessing data
    Xtr = preprocessing.scale(Xtr)
    Xte = preprocessing.scale(Xte)

    Xtr = xgb.DMatrix(Xtr, label=Ytr)
    Xte = xgb.DMatrix(Xte)
    # since we only need the rank
    param = {}
    param['objective'] = 'binary:logitraw'
    # scale weight of positive examples
    sum_wneg=np.sum(Ytr==0)
    sum_wpos=np.sum(Ytr)
    param['scale_pos_weight'] = float(sum_wneg)/float(sum_wpos)
    param['eta'] = 0.1
    param['max_depth'] = 7
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['min_child_weight'] = 100
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['nthread'] = 4
    num_round = 100


    # Training classifier
    print('Fitting started')
    Classifier = xgb.train(param, Xtr, num_round)

    print('Fitting Over')
    print('Starting prediction')



    return (Classifier.predict(Xte)>0)*1


def WriteSubmission(Pred, loc):

    df = pd.DataFrame(Pred)
    df.columns = ['category']
    df.index.name = 'id'
    df.to_csv(loc)
    

    
    
if __name__ == '__main__':
    main()
