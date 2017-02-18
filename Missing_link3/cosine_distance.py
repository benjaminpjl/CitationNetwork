import numpy as np
import nltk
import csv as csv
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances


def cosine_distance():

	stemmer = PorterStemmer()
	nltk.download('stopwords')
	stpwds = set(nltk.corpus.stopwords.words("english"))

	with open("node_information.csv", "r") as f:
		reader = csv.reader(f)
		node_info  = list(reader)

	with open("training_set.txt", "r") as f:
		reader = csv.reader(f)
		training_set  = list(reader)

	training_set = [element[0].split(" ") for element in training_set]

	with open("testing_set.txt", "r") as f:
		reader = csv.reader(f)
		testing_set  = list(reader)

	testing_set = [element[0].split(" ") for element in testing_set]

	def tfidf_abstract():

		tfidf_abstracts = []

		for i in xrange(len(node_info)):
			abstract = node_info[i][5].lower().split(" ")
			abstract = [token for token in abstract if token not in stpwds]
			abstract = [stemmer.stem(token) for token in abstract]
			tfidf_abstracts.append(" ".join(abstract))

		vectorizer = TfidfVectorizer(min_df=2)
		tfidf_abstracts = vectorizer.fit_transform(tfidf_abstracts)

		tfidf_abstracts = tfidf_abstracts.toarray()

		return tfidf_abstracts

	tfidf_abstracts = tfidf_abstract()


	training_distance = []
	testing_distance = []

	IDs = [element[0] for element in node_info]

	for i in xrange(len(training_set)):
		source = training_set[i][0]
		target = training_set[i][1]

		index_source = IDs.index(source)
		index_target = IDs.index(target)		

		source_info = tfidf_abstracts[index_source]
		target_info = tfidf_abstracts[index_target]

		training_distance.append(pairwise_distances(source_info, target_info, metric='cosine', n_jobs=1))
		
	training_distance = np.asarray(training_distance).reshape(len(training_set),)


	for i in xrange(len(testing_set)):
		source = testing_set[i][0]
		target = testing_set[i][1]

		index_source = IDs.index(source)
		index_target = IDs.index(target)		

		source_info = tfidf_abstracts[index_source]
		target_info = tfidf_abstracts[index_target]

		testing_distance.append(pairwise_distances(source_info, target_info, metric='cosine', n_jobs=1))
		
	testing_distance = np.asarray(testing_distance).reshape(len(testing_set),)


	return training_distance,testing_distance
