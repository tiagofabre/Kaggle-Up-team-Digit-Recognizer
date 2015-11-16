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
from sklearn.svm import SVR




print('Get data...')
data = pd.DataFrame.as_matrix(pd.read_csv('train.csv'))
Y = data[:, 0]
data = data[:, 1:] # trim first classification field
X = data

print('Normalize date train...')
X = X/255.0
X[X < 0.1] = 0.0
X[X >= 0.9] = 1.0




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
datateste = (pd.read_csv('test.csv').values).astype('float32')
Z = datateste/255.00

    

###############################################################################
# Generate sample data
X = XX
y = YY

###############################################################################
#print('Add noise to targets')
#y[::5] += 3 * (0.5 - np.random.rand(8))

###############################################################################
print('Fit regression model')
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)


print('Predicting...')
test_data = Z
#test_data = pca.transform(test_data)
predict = svr.predict(test_data)


print('Saving...')
with open('predict.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')







