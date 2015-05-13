import math
import random
import numpy as np
import IPython
import cPickle as pickle 
from numpy import linalg as LA
from sklearn import svm 
from sklearn import preprocessing  
#from scipy.sparse import csr_matrix

class Learner():

	verbose = True

	def Load(self):
		self.States = pickle.load(open('states.p','rb'))
		self.Actions = pickle.load(open('actions.p','rb'))
		self.trainModel(self.States,self.Actions)


	def trainModel(self,States,Action):
		self.clf = svm.SVC()
		self.novel = svm.OneClassSVM()
	
		print States.shape
		print Action.shape
		self.scaler = preprocessing.StandardScaler().fit(States)
		States = self.scaler.transform(States)
		Action = np.ravel(Action)
		
		#States = csr_matrix(States)
		#self.clf.class_weight = 'auto'

		self.novel.nu = 1e-3
		self.novel.gamma = 1e-3
		#self.kde = KernelDensity(kernel = 'gaussian', bandwidth=0.8).fit(States)
		self.clf.fit(States,Action)
		self.novel.fit(States)
		if(self.verbose):
			self.debugPolicy(States,Action)
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

		for i in range(self.getNumData()):
			if(Action[i] not in classes):
				value = np.zeros(3)
				classes.update({Action[i]:value})
			classes[Action[i]][0] += 1
			if(Action[i] != prediction[i]):
				classes[Action[i]][1] += 1

			classes[Action[i]][2] = classes[Action[i]][1]/classes[Action[i]][0] 
		for d in classes:
			print d, classes[d]



 	def getAction(self,state):
 		state = self.scaler.transform(state)
 		#state = csr_matrix(state)
		return self.clf.predict(state)

	def askForHelp(self,state):
		#IPython.embed()
		#if(abs(state[1]) > 80  and abs(state[2]) == 0):
			#IPython.embed() 
			#return -1
		#else: 
			#return 1
		
		state = self.scaler.transform(state)
		#state = csr_matrix(state)
		return self.novel.predict(state)

	def getNumData(self): 
		return self.Actions.shape[0]

	def newModel(self,states,actions):
		#states = csr_matrix(states)
		#actions = csr_matrix(actions)
		self.States = states
		self.Actions = actions
		self.trainModel(self.States,self.Actions)

	def updateModel(self,new_states,new_actions):
		print "UPDATING MODEL"

		#self.States = new_states
		#self.Actions = new_actions
		self.States = np.vstack((self.States,new_states))
		self.Actions = np.vstack((self.Actions,new_actions))
		self.trainModel(self.States,self.Actions)

	def saveModel(self):
		pickle.dump(self.States,open('states.p','wb'))
		pickle.dump(self.Actions,open('actions.p','wb'))