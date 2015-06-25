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


class Learner():

	verbose = True
	useKMM = False 
	gamma = 1e-3
	gamma_clf = 1e-3
	first_time = True 
	iter_  = 1

	def Load(self,gamma = 1e-3):
		self.States = pickle.load(open('states.p','rb'))
		self.Actions = pickle.load(open('actions.p','rb'))
		self.Weights = np.zeros(self.Actions.shape)+1
		self.gamma = gamma 
		self.trainModel(self.States,self.Actions)
		
	def clearModel(self):
		self.States = pickle.load(open('states.p','rb'))
		self.Actions = pickle.load(open('actions.p','rb')) 
		self.Weights = np.zeros(self.Actions.shape)+1
		self.trainModel(self.States,self.Actions)


	def trainModel(self,States,Action):
		self.clf = svm.LinearSVC()
		self.novel = svm.OneClassSVM()
	
		print States.shape
		print Action.shape
	
		Action = np.ravel(Action)
		
		
		self.clf.class_weight = 'auto'

		self.novel.gamma = self.gamma

		self.clf.C = 1e-2
		#self.clf.kernel = 'linear'
		
		if(self.useKMM):
			self.Weights = np.ravel(self.Weights)
			self.clf.fit(States,Action,self.Weights)
		else:
			self.clf.fit(States,Action)
		#SVM parameters computed via cross validation
	
		if(self.verbose):
			errors = self.debugPolicy(States,Action)

		
		mask = np.ones(self.supStates.shape[0],dtype=bool)
		mask[errors] = False
		supStatesClean = self.supStates[mask]
		#self.kde = KernelDensity(kernel = 'gaussian', bandwidth=0.8).fit(States)
		# IPython.embed()
		# Size = self.supStates.shape 
		# for i in range(Size[0]):
		# 	for j in range(Size[1]):
		# 		print i
		# 		if(self.supStates[i,j] != 0 and self.supStates[i,j] != 1):
		# 			print "Incorect", self.supStates[i,j]
		# 			IPython.embed()

		self.novel.nu = 1e-3
		self.novel.kernel = 'rbf'
		self.novel.verbose = False
		self.novel.shrinking = False
		self.novel.max_iter = 3000
		

		self.novel.fit(supStatesClean)
		print self.novel.gamma
		
		
		#self.
		#IPython.embed()
		#IPython.embed()

	def getScoreNovel(self,States):
		num_samples = States.shape[0]
		avg = 0
		for i in range(num_samples):
			ans = self.novel.predict(States[i,:])
			if(ans == -1): 
				ans = 0
			avg = avg+ans/num_samples

		return avg

	def debugPolicy(self,States,Action):
		prediction = self.clf.predict(States)
		classes = dict()
		errors = []

		for i in range(self.getNumData()):
			if(Action[i] not in classes):
				value = np.zeros(3)
				classes.update({Action[i]:value})
			classes[Action[i]][0] += 1
			if(Action[i] != prediction[i]):
				classes[Action[i]][1] += 1

				errors.append(i)

			classes[Action[i]][2] = classes[Action[i]][1]/classes[Action[i]][0] 
		for d in classes:
			print d, classes[d]

		self.precision = self.clf.score(States,Action)

		return errors

	def getPrecision(self):
		return self.precision

 	def getAction(self,state):
 		state = csr_matrix(state)
		return self.clf.predict(state)

	def askForHelp(self,state):
		#IPython.embed()
		#if(abs(state[1]) > 80  and abs(state[2]) == 0):
			#IPython.embed() 
			#return -1
		#else: 
			#return 1
		
		if(isinstance(state,csr_matrix)):
			state = state.todense()

		#state = preprocessing.normalize(state,norm='l2')
		return self.novel.predict(state)

	def getNumData(self): 
		return self.Actions.shape[0]

	def newModel(self,states,actions):
		states = csr_matrix(states)

		self.States = states
		self.supStates = states.todense() 
		self.Actions = actions
		self.Weights = np.zeros(actions.shape)+1
		self.trainModel(self.States,self.Actions)

	def updateModel(self,new_states,new_actions,kmm_state):
		print "UPDATING MODEL"

		#self.States = new_states
		#self.Actions = new_actions
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
		pickle.dump(self.States,open('states.p','wb'))
		pickle.dump(self.Actions,open('actions.p','wb'))
