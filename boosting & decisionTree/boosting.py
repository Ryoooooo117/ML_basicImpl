import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################

		# print('boosting predict')
		H = np.zeros(len(features))
		for i in range(len(self.clfs_picked)):
			clf = self.clfs_picked[i]
			beta = self.betas[i]
			h_t = np.asarray(clf.predict(features))
			H += beta * h_t

		H = np.where(H > 0, 1, -1)
		return H.tolist()
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################

		features_Mat = np.asarray(features)
		N, D = features_Mat.shape
		weights = np.array([1/len(features) for i in range(len(features))]).tolist()

		# predicts = np.array([clf.predict(features) for clf in self.clfs])
		# print('boosting train weights: ',weights)
		# print('boosting train predicts', predicts)
		# print('boosting train labels', labels)
		# errors = np.array([y_i != y_p for y_i, y_p in zip(labels, predicts)])
		# agreements = [-1 if e else 1 for e in errors]

		for t in range(self.T):
			# h = np.argmin([weights[t]*e for e in errors])
			# epsilon = np.sum(errors * weights)
			# beta = 0.5 * np.log((1-epsilon)/epsilon)
			# z = 2 * np.sqrt(epsilon * (1-epsilon))
			# weights = np.array([(weight / z) * np.exp(-1 * beta * agreement) for weight, agreement in zip(weights, agreements)])
			# self.betas.append(beta)
			epsilon = sum(weights)
			h_t = None
			pred = None

			# find h_t
			for clf in self.clfs:
				currPred = clf.predict(features)
				currEps = sum([weights[i] for i in range(len(features)) if labels[i] != currPred[i]])
				if currEps < epsilon:
					epsilon = currEps
					h_t = clf
					pred = currPred

			beta = 0.5 * np.log((1 - epsilon) / epsilon)
			weight_next = [weights[i]*np.exp(-1*beta) if labels[i] == pred[i] else weights[i]*np.exp(beta) for i in range(len(features))]
			weight_next =  weight_next / sum(weight_next)

			self.clfs_picked.append(h_t)
			self.betas.append(beta)
			weights = weight_next

		# print('adaboost train self.betas:',beta)

		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	