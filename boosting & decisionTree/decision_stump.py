import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
	def __init__(self, s:int, b:float, d:int):
		self.clf_name = "Decision_stump"
		self.s = s
		self.b = b
		self.d = d

	def train(self, features: List[List[float]], labels: List[int]):
		pass
		
	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		##################################################
		# TODO: implement "predict"
		##################################################

		# print('DecisionStump: s:', self.s,' b:',self.b, ' d:',self.d, ' x[d]: ',features)
		features = np.asarray(features)
		x_d = features[:, self.d]
		pred = self.s * np.where(x_d > self.b, 1, -1)
		# print('DecisionStump b', self.b, ' x_d ', x_d, ' pred',pred)
		return pred.tolist()
		