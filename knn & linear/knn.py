"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is compute_distances, predict_labels, compute_accuracy
and find_best_k.
"""

import numpy as np
import json


###### Q5.1 ######
def compute_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					                    #

	dists = []
	for xtext in X:
		temp = []
		for xtrain in Xtrain:
			temp.append(np.sqrt(np.sum((xtrain - xtext)**2)))
		dists.append(temp)
	dists = np.array(dists)

	#####################################################

	return dists

###### Q5.2 ######
def predict_labels(k, ytrain, dists):
	"""
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i]. 
	"""
	#####################################################
	#				 YOUR CODE HERE					                    #

	# dists = np.empty([0, 20])
	# for i in range(5):
	# 	distsCol = []
	# 	for j in range(20):
	# 		distsCol.append(np.random.randint(0,100))
	# 	dists = np.vstack((dists, distsCol))

	ypred = list()
	nearestMatrix = list()
	for data in dists:
		nearestArray = []
		sortDistsRow = list(data)
		sortDistsRow.sort()
		# print('sortDistsRow  ', sortDistsRow)
		for j in range(k):
			nearestIndex = list(data).index(sortDistsRow[j])
			# print('nearestIndex  ', nearestIndex)
			nearestClass = ytrain[nearestIndex]
			nearestArray.append(nearestClass)
		# print('     index: ',nearestIndex)
		nearestMatrix.append(nearestArray)

	for i in nearestMatrix:
		counts = np.bincount(i)
		nearestClass = np.argmax(counts)
		ypred.append(nearestClass)
		# print('nearestMatrix: ', i, ' class:',nearestClass)
	# ypredKnearestMatrix = list()
	# for index, indexArray in enumerate(nearestIndexMatrix):
	# 	# for i in indexArray:
	# 	# 	ypredKnearestMatrix.append()
	# 	print(index,' ',indexArray)

	# print('ypred: ',ypred)
	#####################################################
	ypred = np.asarray(ypred)
	return ypred

###### Q5.3 ######
def compute_accuracy(y, ypred):
	"""
	Compute the accuracy of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the 
	  prediction of the ith test point.
	Returns:
	- acc: The accuracy of prediction (scalar).
	"""
	#####################################################
	#				 YOUR CODE HERE					                    #
	#####################################################

	examples = y.shape[0]
	correctExamples = 0
	for i in range(examples):
		if y[i] == ypred[i]:
			correctExamples += 1

	acc = correctExamples / examples
	return acc

###### Q5.4 ######
def find_best_k(K, ytrain, dists, yval):
	"""
	Find best k according to validation accuracy.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the highest validation accuracy.
	- validation_accuracy: A list of accuracies of different ks in K.
	"""
	
	#####################################################
	#				 YOUR CODE HERE					                    #
	#####################################################

	validation_accuracy = list()
	print('ks',K)
	best_k = K[0]
	best_acc = 0
	for k in K:
		ypred = predict_labels(k, ytrain, dists)
		acc = compute_accuracy(yval, ypred)
		if acc > best_acc:
			best_acc = acc
			best_k = k
		validation_accuracy.append(acc)

	print('best k:',best_k)
	print('validation_accuracy:', validation_accuracy)

	return best_k, validation_accuracy


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_processing(data):
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]
	
	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)
	
	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)

	# print('Xtrain: ',Xtrain)
	# print('ytrain: ', ytrain)
	# print('Xval: ', Xval)
	# print('yval: ', yval)
	# print('Xtest: ', Xtest)
	# print('ytest: ', ytest)

	return Xtrain, ytrain, Xval, yval, Xtest, ytest
	
def main():
	input_file = 'mnist_subset.json'
	output_file = 'knn_output.txt'

	with open(input_file) as json_data:
		data = json.load(json_data)
	
	#==================Compute distance matrix=======================
	K=[1, 3, 5, 7, 9]	
	
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	
	dists = compute_distances(Xtrain, Xval)
	
	#===============Compute validation accuracy when k=5=============
	k = 5
	ypred = predict_labels(k, ytrain, dists)
	acc = compute_accuracy(yval, ypred)
	print("The validation accuracy is", acc, "when k =", k)

	# #==========select the best k by using validation set==============
	best_k,validation_accuracy = find_best_k(K, ytrain, dists, yval)


	# #===============test the performance with your best k=============
	dists = compute_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_accuracy = compute_accuracy(ytest, ypred)
	#
	# #====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_accuracy[i])+'\n')
	f.write('%s %.3f' % ('test', test_accuracy))
	f.close()
	
if __name__ == "__main__":
	main()
