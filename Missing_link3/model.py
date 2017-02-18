import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
import csv as csv


with open("X_train.txt", "r") as f:
	reader = csv.reader(f)
	X_train  = list(reader)

X_train = [element[0].split(" ") for element in X_train]

X = []
for i in xrange(len(X_train)):
	X.append([float(j) for j in X_train[i]])

with open("y_train.txt","r") as f:
	reader = csv.reader(f)
	y_train = list(reader)

y_train = np.asarray(y_train).reshape(len(y_train),)
X_train = np.asarray(X).reshape(len(X),len(X[0]))

# Random Forest
for i in np.linspace(10, 100, num = 10).tolist():
	clf = RandomForestClassifier(n_estimators=int(i))
	scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=3)

	print 'trees: ', i, ' score: ', np.mean(scores)

'''
trees:  10.0  score:  0.959687218858
trees:  20.0  score:  0.96062140167
trees:  30.0  score:  0.961045438451
trees:  40.0  score:  0.961396365762
trees:  50.0  score:  0.961300510939
trees:  60.0  score:  0.961232275159
trees:  70.0  score:  0.961396365696
trees:  80.0  score:  0.961527963566 ##
trees:  90.0  score:  0.961399615825
trees:  100.0  score:  0.961490596734
'''

for i in np.linspace(150, 500, num = 8).tolist():
	clf = RandomForestClassifier(n_estimators=int(i))
	scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=3)

	print 'trees: ', i, ' score: ', np.mean(scores)

'''
trees:  150.0  score:  0.961609196537
trees:  200.0  score:  0.961661185762
trees:  250.0  score:  0.96159619928
trees:  300.0  score:  0.961615695253
trees:  350.0  score:  0.961662810431 ##
trees:  400.0  score:  0.961659561259
'''

# Extremely Randomized Trees
for i in np.linspace(10, 100, num = 10).tolist():
	clf = ExtraTreesClassifier(n_estimators=int(i))
	scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=3)

	print 'trees: ', i, ' score: ', np.mean(scores)

'''
trees:  10.0  score:  0.956704336925
trees:  20.0  score:  0.958413483269
trees:  30.0  score:  0.958876512477
trees:  40.0  score:  0.959134833522
trees:  50.0  score:  0.95921606663
trees:  60.0  score:  0.959185197824
trees:  70.0  score:  0.959347664801
trees:  80.0  score:  0.959342790541
trees:  90.0  score:  0.959352538372
trees:  100.0  score:  0.959474388213 ##
'''

# AdaBoost + Random Forest
'''
for i in np.linspace(100, 400, num = 7).tolist():
	clf = AdaBoostClassifier(base_estimator = RandomForestClassifier(n_estimators = 80), n_estimators=int(i))
	scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=3)

	print 'estimators: ', i, ' score: ', np.mean(scores)
'''

X1, X2, y1, y2 = cross_validation.train_test_split(X_train, y_train, test_size=0.4, random_state=0)

clf = AdaBoostClassifier(base_estimator = RandomForestClassifier(n_estimators = 350), n_estimators=50).fit(X1,y1)
print clf.score(X2,y2)
