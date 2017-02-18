import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.stem.porter import PorterStemmer
from os.path import isfile
import networkx as nx
import re
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier


def write_nb_comm_ngbrs(f, x):
    try:
        f.write(str(len(list(nx.common_neighbors(G, x['id1'], x['id2'])))) + '\n')
    except nx.NetworkXError:
        f.write(str(0) + '\n')

def get_nb_comm_ngbrs(row):
    try:
        return len(list(nx.common_neighbors(G, row['id1'], row['id2'])))
    except nx.NetworkXError:
        return 0

def get_degree(row):
    if not G.has_node(row['id1']) or not G.has_node(row['id2']):
        return Series([0, 0])
    try:
        return Series([G.degree(row['id1']), G.degree(row['id2'])])
    except nx.NetworkXError:
        return Series([0, 0])


def get_baseline_features(id1, id2):
    node1 = node_info.ix[id1]
    node2 = node_info.ix[id2]
    # nb of common authors
    authors1 = node1['authors']
    authors2 = node2['authors']
    if authors1 is np.nan or authors2 is np.nan:
        nb_comm_authors = 0
    else:
        authors1 = set(authors1.split(','))
        authors2 = set(authors2.split(','))
        nb_comm_authors = len(authors1.intersection(authors2))

    # nb of common words in title
    title1 = node1['title']
    title2 = node2['title']
    if title1 is np.nan or title2 is np.nan:
        nb_comm_titles_words = 0
    else:
        title1 = title1.lower()
        title2 = title2.lower()
        title1 = title1.split(' ')
        title2 = title2.split(' ')
        title1 = filter(lambda x: x not in stpwds, title1)
        title2 = filter(lambda x: x not in stpwds, title2)
        title1 = map(lambda x: stemmer.stem(x), title1)
        title2 = map(lambda x: stemmer.stem(x), title2)
        nb_comm_titles_words = len(set(title1).intersection(set(title2)))

    # diff of publish year
    year1 = node1['year']
    year2 = node2['year']
    if year1 is np.nan or year2 is np.nan:
        diff_year = 0
    else:
        diff_year = np.abs(year1 - year2)
    return Series([nb_comm_titles_words, diff_year, nb_comm_authors])

def get_nb_comm_journal_words(id1, id2):
    node1 = node_info.ix[id1]
    node2 = node_info.ix[id2]
    journal1 = node1['journal']
    journal2 = node2['journal']
    if journal1 is np.nan or journal2 is np.nan:
        return 0 
    else:
        journal1 = journal1.lower()
        journal2 = journal2.lower()
        # split by punctuation and then remove empty string
        journal1 = filter(None, re.split('\W+', journal1))
        journal2 = filter(None, re.split('\W+', journal2))
        return len(set(journal1).intersection(set(journal2)))

def get_abstract_simi(id1, id2):
    vec1 = vec[id1]
    vec2 = vec[id2]
    return linear_kernel(vec1, vec2)[0][0]

def get_path_length(row):
    if not G.has_node(row['id1']) or not G.has_node(row['id2']):
        return -1
    try:
        return len(nx.shortest_path(G, source = row['id1'], target = row['id2']))
    except nx.NetworkXError:
        return -1
    except nx.NetworkXNoPath:
        return -1



if __name__ == '__main__':
    ## nltk
    print 'loading nltk...'
    stemmer = PorterStemmer()
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words('english'))

    ## read node info
    print 'reading node info...'
    node_info = pd.read_csv('node_info.csv', sep = ',', header = None, names = ['id_', 'year', 'title', 'authors', 'journal', 'abstract'])
    node_info.set_index('id_', inplace = True)

    ## read training set
    print 'reading train data...'
    training_set = pd.read_csv('training_set.txt', sep = ' ', header = None, names = ['id1', 'id2', 'is_edge'])

    ## read testing set
    print 'reading test data'
    testing_set = pd.read_csv('testing_set.txt', sep = ' ', header = None, names = ['id1', 'id2'])

    X_train = training_set.drop('is_edge', axis = 1)
    y_train = training_set['is_edge']
    X_test = testing_set[['id1', 'id2']]

    ## feature: baseline features
    print 'extracting baseline features...train data...'
    baseline_features_train_file = 'baseline_features_train.csv'
    baseline_features_test_file = 'baseline_features_test.csv'
    if isfile(baseline_features_train_file):
        baseline_features_train = pd.read_csv(baseline_features_train_file)
    else:
        baseline_features_train = training_set.apply(lambda x: get_baseline_features(x['id1'], x['id2']), axis = 1)
        baseline_features_train.columns = ['nb_comm_titles_words', 'diff_year', 'nb_comm_authors']
        baseline_features_train.to_csv(baseline_features_train_file, index = False)
    print 'extracting baseline features...test data...'
    if isfile(baseline_features_test_file): 
        baseline_features_test = pd.read_csv(baseline_features_test_file)
    else:
        baseline_features_test = testing_set.apply(lambda x: get_baseline_features(x['id1'], x['id2']), axis = 1)
        baseline_features_test.columns = ['nb_comm_titles_words', 'diff_year', 'nb_comm_authors']
        baseline_features_test.to_csv('baseline_features_test.csv', index = False)
    baseline_features_train.set_index(training_set.index)
    baseline_features_test.set_index(testing_set.index)
    X_train = pd.concat([X_train, baseline_features_train], axis = 1)
    X_test = pd.concat([X_test, baseline_features_test], axis = 1)

    ## feature: nb of common words in journal
    nb_comm_journal_words_train_file = 'nb_comm_journal_words_train.csv'
    nb_comm_journal_words_test_file = 'nb_comm_journal_words_test.csv'
    print 'extracting nb_comm_journal_words... train'
    if isfile(nb_comm_journal_words_train_file):
        nb_comm_journal_words_train = pd.read_csv(nb_comm_journal_words_train_file)
    else:
        nb_comm_journal_words_train = training_set.apply(lambda x: get_nb_comm_journal_words(x['id1'], x['id2']), axis = 1)
        nb_comm_journal_words_train.name = 'nb_comm_journal_words'
        nb_comm_journal_words_train = nb_comm_journal_words_train.to_frame()
        nb_comm_journal_words_train.to_csv(nb_comm_journal_words_train_file, index = False, header = True)
    print 'extracting nb_comm_journal_words... test'
    if isfile(nb_comm_journal_words_test_file):
        nb_comm_journal_words_test = pd.read_csv(nb_comm_journal_words_test_file)
    else:
        nb_comm_journal_words_test = testing_set.apply(lambda x: get_nb_comm_journal_words(x['id1'], x['id2']), axis = 1)
        nb_comm_journal_words_test.name = 'nb_comm_journal_words'
        nb_comm_journal_words_test = nb_comm_journal_words_test.to_frame()
        nb_comm_journal_words_test.to_csv(nb_comm_journal_words_test_file, index = False, header = True)
    nb_comm_journal_words_train.set_index(training_set.index)
    nb_comm_journal_words_test.set_index(testing_set.index)
    X_train = pd.concat([X_train, nb_comm_journal_words_train], axis = 1)
    X_test = pd.concat([X_test, nb_comm_journal_words_test], axis = 1)

    ## extracting introduction similarities
    abstract_simi_train_file = 'abstract_simi_train.csv'
    abstract_simi_test_file = 'abstract_simi_test.csv'
    print 'calculating abstract similarities... train'
    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = stpwds, norm = 'l2')
    vec = vectorizer.fit_transform(node_info['abstract'])
    if isfile(abstract_simi_train_file):
        abstract_simi_train = pd.read_csv(abstract_simi_train_file)
    else:
        abstract_simi_train = training_set.apply(lambda x: get_abstract_simi(node_info.index.get_loc(x['id1']), node_info.index.get_loc(x['id2'])), axis = 1)
        abstract_simi_train.name = 'abstract_similarities'
        abstract_simi_train = abstract_simi_train.to_frame()
        abstract_simi_train.to_csv(abstract_simi_train_file, index = False, header = True)

    print 'calculating abstract similarities... test'
    if isfile(abstract_simi_test_file):
        abstract_simi_test = pd.read_csv(abstract_simi_test_file)
    else:
        abstract_simi_test = testing_set.apply(lambda x: get_abstract_simi(node_info.index.get_loc(x['id1']), node_info.index.get_loc(x['id2'])), axis = 1)
        abstract_simi_test.name = 'abstract_similarities'
        abstract_simi_test = abstract_simi_test.to_frame()
        abstract_simi_test.to_csv(abstract_simi_test_file, index = False, header = True)

    abstract_simi_train.set_index(training_set.index)
    abstract_simi_test.set_index(testing_set.index)
    X_train = pd.concat([X_train, abstract_simi_train], axis = 1)
    X_test = pd.concat([X_test, abstract_simi_test], axis = 1)

    
    ## filter those without real edges
    training_set_edge = training_set[training_set['is_edge'] == 1]
    ## initialize graph
    G = nx.Graph()
    ## fill graph edges
    print 'filling graph...'
    training_set_edge.apply(lambda x: G.add_edge(x['id1'], x['id2']), axis = 1)
    del training_set_edge

    ## feature: nb of common neighbours
    nb_comm_ngbrs_train_file = 'nb_comm_ngbrs_train.csv'
    nb_comm_ngbrs_test_file = 'nb_comm_ngbrs_test.csv'
    print 'extracting nb of common neighbours... train'
    if isfile(nb_comm_ngbrs_train_file):
        nb_comm_ngbrs_train = pd.read_csv(nb_comm_ngbrs_train_file)
    else:
        nb_comm_ngbrs_train = training_set.apply(get_nb_comm_ngbrs, axis = 1) 
        nb_comm_ngbrs_train.name = 'nb_comm_ngbrs'
        nb_comm_ngbrs_train = nb_comm_ngbrs_train.to_frame()
        nb_comm_ngbrs_train.to_csv(nb_comm_ngbrs_train_file, index = False, header = True)
    print 'extracting nb of common neighbours... test'
    if isfile(nb_comm_ngbrs_test_file):
        nb_comm_ngbrs_test = pd.read_csv(nb_comm_ngbrs_test_file)
    else:
        nb_comm_ngbrs_test = testing_set.apply(get_nb_comm_ngbrs, axis = 1) 
        nb_comm_ngbrs_test.name = 'nb_comm_ngbrs'
        nb_comm_ngbrs_test = nb_comm_ngbrs_test.to_frame()
        nb_comm_ngbrs_test.to_csv(nb_comm_ngbrs_test_file, index = False, header = True)
    nb_comm_ngbrs_train.set_index(training_set.index)
    nb_comm_ngbrs_test.set_index(testing_set.index)
    X_train = pd.concat([X_train, nb_comm_ngbrs_train], axis = 1)
    X_test = pd.concat([X_test, nb_comm_ngbrs_test], axis = 1)

    ## feature: node degree
    degree_train_file = 'degree_train.csv'
    degree_test_file = 'degree_test.csv'
    print 'extracting node degrees... train'
    if isfile(degree_train_file):
        degree_train = pd.read_csv(degree_train_file)
    else:
        degree_train = training_set.apply(get_degree, axis = 1) 
        degree_train.columns = ['degree1', 'degree2']
        degree_train.to_csv(degree_train_file, index = False, header = True)
    print 'extracting node degrees... test'
    if isfile(degree_test_file):
        degree_test = pd.read_csv(degree_test_file)
    else:
        degree_test = testing_set.apply(get_degree, axis = 1) 
        degree_test.columns = ['degree1', 'degree2']
        degree_test.to_csv(degree_test_file, index = False, header = True)
    degree_train.set_index(training_set.index)
    degree_test.set_index(testing_set.index)
    X_train = pd.concat([X_train, degree_train], axis = 1)
    X_test = pd.concat([X_test, degree_test], axis = 1)

    ### feature: length of shortest path
    # path_train_file = 'path_train.csv'
    # path_test_file = 'path_test.csv'
    # print 'extracting length of shortest path... train'
    # if isfile(path_train_file):
    #     path_train = pd.read_csv(path_train_file)
    # else:
    #     path_train = training_set.apply(get_path_length, axis = 1) 
    #     path_train.name = 'length_shortest_path'
    #     path_train = path_train.to_frame()
    #     path_train.to_csv(path_train_file, index = False, header = True)
    # print 'extracting length of shortest path... test'
    # if isfile(path_test_file):
    #     path_test = pd.read_csv(path_test_file)
    # else:
    #     path_test = testing_set.apply(get_path_length, axis = 1) 
    #     path_test.name = 'length_shortest_path'
    #     path_test = path_test.to_frame()
    #     path_test.to_csv(path_test_file, index = False, header = True)
    # path_train.set_index(training_set.index)
    # path_test.set_index(testing_set.index)
    # X_train = pd.concat([X_train, path_train/14], axis = 1)
    # X_test = pd.concat([X_test, path_test/14], axis = 1)

    ## assembling all features
    X_train = X_train.drop(['id1', 'id2'], axis = 1)
    X_test = X_test.drop(['id1', 'id2'], axis = 1)
    y_train = np.asarray(y_train, dtype = int)
    ## transform to 32 bit float for GPU acceleration
    X_train = np.asarray(X_train, dtype = 'float32')
    X_test = np.asarray(X_test, dtype = 'float32')

    # fit and prediction
    print "fit and prediction_GradientBoostingClassifier"
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    y_predict = DataFrame(y_predict)
    y_predict.index.name = 'id'
    y_predict.columns = ['category']
    y_predict.to_csv('predictions.csv')
