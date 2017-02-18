import numpy as np
from sklearn.cross_validation import ShuffleSplit
import networkx as nx
import csv as csv

class graph_classifer(object):

	def __init__(self):
		pass

	def train_graph(self,G_tr):
		with open("G_tr.txt","wb") as f:
			writer = csv.writer(f)
			for i in G_tr:
				if i[0].split(' ')[2] == '1':
					writer.writerow(i)

		self.G = nx.read_weighted_edgelist("G_tr.txt")

		for (u,v,d) in self.G.edges(data = True):
			if d['weight'] == 0:
				d['weight'] = 27770

	# def predict(self,to_test):

	# 	# def path_weight(path):
	# 	# 	weight = []
	# 	# 	start = path[0]
	# 	# 	for i in range(len(path)-1):
	# 	# 		weight.append(int(self.G[start][path[i+1]]['weight']))
	# 	# 		start = path[i+1]
	# 	# 	return sum(weight)

	# 	results = []
	# 	ct = len(to_test)-1
	# 	while ct != -1:
	# 		try:
	# 			path_len = nx.dijkstra_path_length(self.G,to_test[ct][0],to_test[ct][1])
	# 			if path_len > 2:
	# 				results.append(0)
	# 				ct -= 1
	# 				print ct,'0'
	# 			else:
	# 				results.append(1)
	# 				ct -= 1
	# 				print ct,'1'

	# 		except nx.NetworkXNoPath:
	# 			results.append(0)
	# 			ct -= 1
	# 			print ct,'how about that'
	# 		except KeyError:
	# 			results.append(0)
	# 			ct -= 1
	# 			print "node doesn't exist in trained graph"
	# 	return results

	def predict(self,to_test):
		results = []
		path = nx.all_pairs_dijkstra_path_length(self.G,cutoff=2)
		ct = len(to_test)-1
		while ct != -1:
			try:
				path[to_test[ct][0]][to_test[ct][1]]
				results.append(1)
				ct -= 1
				print ct,'1'
			except nx.NetworkXNoPath:
				results.append(0)
				ct -= 1
				print ct,'no path found'
			except KeyError:
				results.append(0)
				ct -= 1
				print ct,"node doesn't exist in trained graph"
		return results


with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

rs = ShuffleSplit(len(training_set),n_iter=1,test_size=0.2, random_state=0)
for tr, te in rs:
	G_tr = [training_set[i] for i in tr]
	G_te_raw = [training_set[i] for i in te]
G_te = []
for i in G_te_raw:
	G_te.append(i[0].split(" "))
G_te = np.asarray(G_te)

to_test = G_te[:,:2]
true_value = G_te[:,2:]

G = graph_classifer()
G.train_graph(G_tr)
results = G.predict(to_test)




