import math
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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


class KMM():

	B = 1000
	def assembleKernel(self,data,data_p):
		self.m = data.shape[0]
		self.m_prime = data_p.shape[0]
		gamma = 1e-4		

		self.K = euclidean_distances(data,data)
		
		self.K = np.exp((self.K**2)*-gamma)

		data_concat = vstack((data,data_p))
		self.Big_K = euclidean_distances(data_concat,data_concat)
		
		self.Big_K = np.exp((self.Big_K**2)*-gamma)
		
		self.k = np.zeros((self.m,1))

		self.epsilon = (self.m**0.5-1)/self.m**0.5
		
		for i in range(self.m):
			self.k[i] = np.sum(self.Big_K[i,self.m:self.Big_K.shape[1]])
			self.k[i] = self.m/self.m_prime * self.k[i]
		self.k = -self.k


	def solveQP(self):
		P = matrix(self.K)
		q = matrix(self.k)

		h = np.zeros((self.m*2+2,1))
		h[0] = self.m*self.epsilon+self.m
		h[1] = self.m*self.epsilon-self.m 
		h[2:self.m+2] = self.B

		G = np.zeros((1,self.m)) + 1
		G = np.vstack((G,np.zeros((1,self.m))-1))
		G = np.vstack((G,np.eye(self.m)))
		G = np.vstack((G,np.eye(self.m)*-1))
	
		h = matrix(h)
		G = matrix(G)
	

		sol = solvers.qp(P,q,G,h)
		weights = np.array(sol['x'])
		# plt.figure(3)
		# plt.plot(weights)
		# plt.show()
		
		
		return weights