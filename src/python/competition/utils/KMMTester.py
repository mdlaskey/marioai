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
from KMM import KMM

DIM = 5

mean = np.zeros(DIM)
mean_p = np.zeros(DIM)+10

cov = np.eye(DIM)
cov_p = np.eye(DIM)

data = np.random.multivariate_normal(mean,cov,300)
data_p = np.random.multivariate_normal(mean_p,cov,100)

kmm = KMM()

kmm.assembleKernel(data,data_p)

weights = kmm.solveQP()

IPython.embed()