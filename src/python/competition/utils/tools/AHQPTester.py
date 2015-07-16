import random
import numpy as np
import IPython
import cPickle as pickle 
from numpy import linalg as LA
from sklearn import svm 
from sklearn import preprocessing  
from sklearn import linear_model
from sklearn import metrics 
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.spatial.distance import pdist,squareform
from cvxopt import matrix, solvers
from AHQP import AHQP
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.font_manager

import time


def test_sparse_implementations(ahqp, data, labels):
	print "Original"
	start = time.time()
	a = ahqp.assembleKernel(data,labels)
	end = time.time()
	print end - start

	print "Implementation 1"
	start = time.time()
	b = ahqp.assembleKernelSparse1(data,labels)
	end = time.time()
	print end - start

	print "Implementation 2"
	start = time.time()
	c = ahqp.assembleKernelSparse2(data,labels)
	end = time.time()
	print end - start

	print a, b, c

DIM = 2

mean = np.zeros(DIM)
mean_p = np.zeros(DIM)+10

cov = np.eye(DIM)
cov_p = np.eye(DIM)


data = pickle.load(open('states.p','rb'))
#data = np.random.multivariate_normal(mean,cov,10)
data = data[:,1:3]
DIM = data.shape[0]
labels = np.zeros((DIM,1))+1.0
#IPython.embed()
for i in range(data.shape[0]):
	if((data[i,0]<200.0 and data[i,1] > 800) or (data[i,0]>800.0 and data[i,1] < 400.0)):
		labels[i] = -1.0

ahqp = AHQP()

#ahqp.assembleKernel(data,labels)

test_sparse_implementations(ahqp, data, labels)


weights = ahqp.solveQP(DIM)


# Learn a frontier for outlier detection with several classifiers
xx1, yy1 = np.meshgrid(np.linspace(1400,0, 50), np.linspace(1400, 0, 50))
# xx1, yy1 = np.meshgrid(np.linspace(-4,4), np.linspace(-4, 4))

plt.figure(1)


test = np.c_[xx1.ravel(), yy1.ravel()]
Z1 = np.zeros((test.shape[0],1))
for i in range(test.shape[0]):
	Z1[i] = ahqp.predict(test[i,:])

# clf.fit(X1)
# Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
Z1 = Z1.reshape(xx1.shape)

plt.figure(1)  # two clusters

plt.contour(
    xx1, yy1, Z1, levels=[0], linewidths=5, colors='r')



# Plot the results (= shape of the data points cloud)
for i in range(DIM):
	if(labels[i] == -1):
		plt.scatter(data[i, 0], data[i, 1], color='green',s=10)
	else:
		plt.scatter(data[i,0],data[i,1],color = 'black',s=10)


plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

plt.ylabel("Y Position")
plt.xlabel("X Position")

plt.show()


#IPython.embed()