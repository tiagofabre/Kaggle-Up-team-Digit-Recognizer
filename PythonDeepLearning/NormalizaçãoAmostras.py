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





USE_PICKLE = False
IMAGE_WIDTH = 28




data = pd.DataFrame.as_matrix(pd.read_csv('train.csv'))
Y = data[:, 0]
data = data[:, 1:] # trim first classification field
X = data
X = X/255.00


nudge_size = 2
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



#def threshold(X):
#    X[X < 0.1] = 0.0
#    X[X >= 0.9] = 1.0
#    return X



#n_rotations=2
#for rot_i in range(n_rotations):
#    rot_shape = (X.shape[0], X.shape[1])
#    rot_X = np.zeros(rot_shape)
#    for index in range(X.shape[0]):
#        sign = random.choice([-1, 1])
#        angle = np.random.randint(1, 12)*sign
#        rot_X[index, :] = threshold(nd.rotate(np.reshape(X[index, :], ((IMAGE_WIDTH, IMAGE_WIDTH))), angle, reshape=False).ravel())
#    XX = np.vstack((X,rot_X))
#    YY = np.hstack((Y,Y))
    

datateste = pd.DataFrame.as_matrix(pd.read_csv('test.csv'))
Z = datateste/255.00



c = csv.writer(open("amostras200.csv", "wb"))
for row in X:
    c.writerow(row)

d = csv.writer(open("senhas.csv", "wb"))
for row in Y:
    d.writerow(row)




