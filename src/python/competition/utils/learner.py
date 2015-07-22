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
from KMM import KMM
from AHQP import AHQP
import sys


import random


class Learner():
	verbose = True
	useKMM = False

	gamma = 1e-3
	gamma_clf = 1e-3
	first_time = False
	iter_ = 1
	use_AHQP = True
	
	def __init__(self,sigma=1.0):
		self.ahqp_solver = AHQP(sigma)

	def Load(self, gamma=1e-3):
		self.States = pickle.load(open('states.p', 'rb'))
		self.Actions = pickle.load(open('actions.p', 'rb'))
		self.Weights = np.zeros(self.Actions.shape) + 1
		self.gamma = gamma
		self.trainModel(self.States, self.Actions)

	def clearModel(self):
		self.States = pickle.load(open('states.p', 'rb'))
		self.Actions = pickle.load(open('actions.p', 'rb'))
		self.Weights = np.zeros(self.Actions.shape) + 1
		self.trainModel(self.States, self.Actions)

	def split_training_test(self, States, Action, TRAIN_SIZE=1000):
		"""
		Splits the states/action pairs into
		training/test sets
		"""
		total_size = len(States)

		train_indices = random.sample([i for i in range(total_size)], TRAIN_SIZE)
		test_indices = [i for i in range(total_size) if i not in train_indices]

		train_states = np.array([np.array(States[i]).astype(np.float32) for i in train_indices])
		train_actions = np.array([Action[i] for i in train_indices]).astype(np.float32)
		test_states = np.array([np.array(States[i]).astype(np.float32) for i in test_indices])
		test_actions = np.array([Action[i] for i in test_indices]).astype(np.float32)

		return train_states, train_actions, test_states, test_actions

	def output_HDF5(self, States, Action):
		"""
		Outputs the given states/actions into
		a HDF5 file for neural net training in Caffe
		"""

		# Creates the data folder and files in the 'net' subdirectory
		train_filename = os.path.join(self.NET_SUBDIR, 'train.h5')
		test_filename = os.path.join(self.NET_SUBDIR, 'test.h5')

		# train/test.txt should be a list of HDF5 files to be read
		with open(os.path.join(self.NET_SUBDIR, 'train.txt'), 'w') as f:
			f.write(train_filename + '\n')
		with open(os.path.join(self.NET_SUBDIR, 'test.txt'), 'w') as f:
			f.write(test_filename + '\n')

		States = np.array(States.toarray())
		Action = np.array(Action)

		train_states, train_actions, test_states, test_actions = self.split_training_test(States, Action)

		# Writing to train/test files
		with h5py.File(train_filename, 'w') as f:
				f['data'] = train_states
				f['label'] = train_actions
		with h5py.File(test_filename, 'w') as f:
				f['data'] = test_states
				f['label'] = test_actions

	def trainModel(self, States, Action):
		"""
		Trains model on given states and actions.
		Uses neural net or SVM based on global
		settings.
		"""
		print "States.shape"
		print States.shape
		print "Action.shape"
		print Action.shape

		Action = np.ravel(Action)

		self.clf = svm.LinearSVC()
		self.novel = svm.OneClassSVM()
		self.clf.C = 1e-2

		#self.clf.kernel = 'linear'
		
		if(self.useKMM):
			self.Weights = np.ravel(self.Weights)
			self.clf.fit(States,Action,self.Weights)
		else:
			self.clf.fit(States,Action)
		#SVM parameters computed via cross validation
	
	
		# self.novel.gamma = self.gamma
		# self.novel.nu = 1e-3
		# self.novel.kernel = 'rbf'
		# self.novel.verbose = False
		# self.novel.shrinking = False
		# self.novel.max_iter = 3000


		# self.novel.fit(supStatesClean)


		if (self.verbose or self.use_AHQP):
			self.debugPolicy(States, Action)


		if self.use_AHQP:
			
			self.ahqp_solver.assembleKernelSparse(States, self.labels)
			self.ahqp_solver.solveQP()
	
		# self.
		# IPython.embed()

	def getScoreNovel(self, States):


		print self.novel.gamma
		
		
		#self.
		#IPython.embed()
		#IPython.embed()
		num_samples = States.shape[0]
		avg = 0
		for i in range(num_samples):
			ans = self.novel.predict(States[i, :])
			if (ans == -1):
				ans = 0
			avg = avg + ans / num_samples

		return avg

	def debugPolicy(self, States, Action):
	
		prediction = self.clf.predict(States)
		classes = dict()
		self.labels = np.zeros(self.getNumData())
		for i in range(self.getNumData()):
			if (Action[i] not in classes):
				value = np.zeros(3)
				classes.update({Action[i]: value})
			classes[Action[i]][0] += 1
			if (Action[i] != prediction[i]):
				classes[Action[i]][1] += 1
				self.labels[i] = -1.0
			else:
				self.labels[i] = 1.0
			classes[Action[i]][2] = classes[Action[i]][1] / classes[Action[i]][0]
		for d in classes:
			print d, classes[d]

		self.precision = self.clf.score(States, Action)

	def getPrecision(self):
		return self.precision

	def getAction(self, state):
		"""
		Returns a prediction given the input state.
		Uses neural net or SVM based on global
		settings.
		"""
		if self.neural:
			net = caffe.Net (self.MODEL_FILE,self.TRAINED_MODEL,caffe.TEST)
			# Caffe takes in 4D array inputs.
			data4D = np.zeros([1,1,27136,1]) 
			# Fill in third dimension
			data4D[0,0,:,0] = state[0,:]
			# Forward call creates a dictionary corresponding to the layers
			pred_dict = net.forward_all(data=data4D)
			# 'prob' layer contains actions and their respective probabilities
			prediction = pred_dict['prob'].argmax()
			return prediction
		else:
			state = csr_matrix(state)
			return self.clf.predict(state)

	def askForHelp(self, state):
		# IPython.embed()
		# if(abs(state[1]) > 80  and abs(state[2]) == 0):
		# IPython.embed()
		# return -1
		# else:
		# return 1

		# state = self.scaler.transform(state)
		
		# state = preprocessing.normalize(state,norm='l2')
		return self.ahqp_solver.predict(csr_matrix(state))

 	def getAction(self,state):
 		state = csr_matrix(state)
		return self.clf.predict(state)

	

	def getNumData(self):
		return self.Actions.shape[0]

	def newModel(self, states, actions):
		states = csr_matrix(states)

		self.States = states
		self.supStates = states.todense()
		self.Actions = actions
		self.Weights = np.zeros(actions.shape) + 1
		self.trainModel(self.States, self.Actions)


	def updateModel(self,new_states,new_actions,kmm_state):

		print "UPDATING MODEL"

		# self.States = new_states
		# self.Actions = new_actions
		new_states = csr_matrix(new_states)

		self.States = vstack((self.States,new_states))
		

		if(self.useKMM):
			kmm = KMM()
			kmm.assembleKernel(self.States,kmm_state)
			self.Weights = kmm.solveQP()
			self.Weights =  np.zeros((self.Weights.shape[0],1))+self.Weights


		
		self.supStates = np.vstack((self.supStates,new_states.todense()))
		self.Actions = np.vstack((self.Actions,new_actions))
		self.trainModel(self.States,self.Actions)


	def saveModel(self):
		pickle.dump(self.States, open('states.p', 'wb'))
		pickle.dump(self.Actions, open('actions.p', 'wb'))
