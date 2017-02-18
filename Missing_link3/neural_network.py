import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from load_data import load_data
from sklearn import cross_validation
import csv as csv


trainX,trainy,testX = load_data()


# the data, shuffled and split between train and test sets
# convert class vectors to binary class matrices
X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainX,trainy, 
									test_size=0.4, random_state=0)
Y_train = np.array([y_train,(1-y_train)]).T
Y_test = np.array([y_test,(1-y_test)]).T


# neural network building...
model = Sequential()
model.add(Dense(64, input_shape=(4,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# choose a optimizer for our neural network
rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train,
          batch_size=128, nb_epoch=10,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# write the predictions to submit
prediction1 = model.predict_classes(testX, batch_size=128, verbose=1)
prediction1 = zip(range(len(testX)),prediction1 )

with open("predictions2.csv","wb") as pred:
    csv_out = csv.writer(pred)
    for row in prediction1:
        csv_out.writerow(row)





