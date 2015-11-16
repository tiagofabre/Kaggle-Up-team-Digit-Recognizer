from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop


import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import csv
import cPickle as pickle
import scipy.ndimage as nd
import pandas as pd
import random
import scipy
import time


import matplotlib.pyplot as plt

import csv
import pandas as pd
import numpy as np

# Setup
np.random.seed(1337)  # for reproducibility
batch_size = 128
nb_epoch = 20

USE_PICKLE = False
IMAGE_WIDTH = 28



print('Get data...')
data = pd.DataFrame.as_matrix(pd.read_csv('train.csv'))
Y = data[:, 0]
data = data[:, 1:] # trim first classification field
X = data

print('Normalize date train...')
X = X/255.0
X[X < 0.1] = 0.0
X[X >= 0.9] = 1.0




def nudge_dataset(X, Y):
    print ('Expand date train...')
    nudge_size = 1
    direction_matricies = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

        

    scaled_direction_matricies = [[[comp*nudge_size for comp in vect] for vect in matrix] for matrix in direction_matricies]
    shift = lambda x, w: convolve(x.reshape((IMAGE_WIDTH, IMAGE_WIDTH)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in scaled_direction_matricies])

    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y



def threshold(X):
    X[X < 0.1] = 0.0
    X[X >= 0.9] = 1.0
    return X


print('Rotation date...')
rot_X = np.zeros(X.shape)
for index in range(X.shape[0]):
    sign = random.choice([-1, 1])
    angle = np.random.randint(8, 16)*sign
    rot_X[index, :] = nd.rotate(np.reshape(X[index, :],
        ((28, 28))), angle, reshape=False).ravel()
XX = np.vstack((X,rot_X))
YY = np.hstack((Y,Y))

    
print('Get and normalize date test...')
datateste = pd.DataFrame.as_matrix(pd.read_csv('test.csv'))
Z = datateste/255.00


# Setup
np.random.seed(1337)  # for reproducibility
batch_size = 128
nb_epoch = 20

# Read Data
print('Reading data...')
labels = YY
X_train = XX
X_test = Z

# pre-processing
#y_train = np_utils.to_categorical(labels)
#scale = np.max(X_train)
#X_train /= scale
#X_test /= scale
#mean = np.std(X_train)
#X_train -= mean
#X_test -= mean
#input_dim = X_train.shape[1]
#nb_classes = y_train.shape[1]

# Cleaning data
#labels = YY[:, 0].values.astype('int32')
#X_train = ([:, 1:].values).astype('float32')
#y_train = np_utils.to_categorical(labels)

# Model
model = Sequential()
model.add(Dense(784, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

print("Training...")
fitlog = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2)

# print(fitlog.history)
# print(fitlog.epoch)
# print(fitlog.totals)

# Plot Results
# plt.plot(fitlog.epoch, fitlog.history['acc'], 'g-')
# plt.plot(fitlog.epoch, fitlog.history['val_acc'], 'g--')
# plt.plot(fitlog.epoch, fitlog.history['loss'], 'r-')
# plt.plot(fitlog.epoch, fitlog.history['val_loss'], 'r--')
# plt.show()

print("Generating test predictions ...")
preds = model.predict_classes(X_test, verbose=1)
print(preds)


def save_to_csv(preds, fname):
    """
    Save the results into a csv file.
    """
    pd.DataFrame({"ImageId": list(range(1, len(preds) + 1)), "Label": preds}).to_csv(fname, index=False, header=True)

# Save results
print('Saving results to .csv')
save_to_csv(preds, "results.csv")

