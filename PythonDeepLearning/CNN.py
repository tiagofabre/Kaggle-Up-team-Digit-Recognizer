import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn import svm, metrics
import numpy as np
import sklearn.decomposition as deco
import pandas as pd
from sklearn import linear_model
from nolearn.dbn import DBN
from scipy.ndimage import convolve
import csv
import cPickle as pickle
import scipy.ndimage as nd
import pandas as pd
import random
import scipy
import time
import os.path



TRAINING_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "train.csv")
TRAINING_SET_PICKLE_PATH = os.path.join(os.path.dirname(__file__), "pickles", "train.p")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "test.csv")
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "data", "knn_benchmark.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "result.csv")



USE_PICKLE = False
IMAGE_WIDTH = 28






def load_training_data():
    print('Get data train/target...')
    data = pd.DataFrame.as_matrix(pd.read_csv('train.csv'))
    Y = data[:, 0]
    data = data[:, 1:] # trim first classification field
    X = normalize_data(data)
    return X, Y

def normalize_data(X):
    print('Normalize date train...')
    X = X/255.0
    return X


def images_to_data(images):
    return np.reshape(images,(len(images),-1))

def average(x):
    return sum(x)/len(x)

def compress_images(images):
    new_images = []
    print images[0]
    for image in images:
        new_image = [[average([image[y*4, x*4], image[y*4, x*4+1], image[y*4+1, x*4], image[y*4+1, x*4+1]]) for x in range(0,28/4)] for y in range(0,28/4)]
        new_images.append(new_image)
    return np.array(new_images)


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

    Y = np.concatenate([Y for _ in range(2)], axis=0)
    return X, Y



def threshold(X):
    X[X < 0.1] = 0.0
    X[X >= 0.9] = 1.0
    return X


def rotate_dataset(X, Y):
    print('Rotation date...')
    rot_X = np.zeros(X.shape)
    for index in range(X.shape[0]):
        sign = random.choice([-1, 1])
        angle = np.random.randint(8, 16)*sign
        rot_X[index, :] = threshold(nd.rotate(np.reshape(X[index, :],
            ((28, 28))), angle, reshape=False).ravel())
    XX = np.vstack((X,rot_X))
    YY = np.hstack((Y,Y))
    return XX, YY
    
print('Get and normalize date test...')
datateste = pd.DataFrame.as_matrix(pd.read_csv('test.csv'))
Z = datateste/255.00

def sigmoid(X):
    return scipy.special.expit(X)

def get_test_data_set():
    data = pd.DataFrame.as_matrix(pd.read_csv('test.csv'))
    X = normalize_data(data)
    return X

def get_benchmark():
    return pd.read_csv(BENCHMARK_PATH)

def get_time_hash():
    return str(int(time.time()))

def make_predictions_path():
    base_string = "predictions"
    file_name = base_string + "-" + get_time_hash() + ".csv"
    file_path = os.path.join(os.path.dirname(__file__), "data", file_name)
    return file_path

def write_predictions_to_csv(predictions):
    csv_path = make_predictions_path()
    predictions_dict = {"ImageId": range(1, len(predictions)+1), "Label": predictions}
    predictions_table = pd.DataFrame(predictions_dict)
    predictions_table.to_csv(csv_path, index=False)




X_train, Y_train = load_training_data()

X_train, Y_train = rotate_dataset(X_train, Y_train)
#X_train, Y_train = nudge_dataset(X_train, Y_train)

n_features = X_train.shape[1]
n_classes = 10
classifier = DBN([n_features, 10, n_classes], 
    learn_rates=0.01, learn_rate_decays=0.9 ,epochs=1, verbose=1)

classifier.fit(X_train, Y_train)

test_data = Z
predictions = classifier.predict(test_data)
csv_path = make_predictions_path()
write_predictions_to_csv(predictions)


def __main__(args):
    run()

if __name__ == "__main__":
    __main__(sys.argv)








