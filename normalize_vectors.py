from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from sklearn import preprocessing

def normalize(train_vector, test_vector, normalization_approach):

		
	if normalization_approach=='znorm':
		normalized_train_vector=(train_vector-train_vector.mean())/train_vector.std()
		normalized_test_vector=(test_vector-train_vector.mean())/train_vector.std()
	if normalization_approach=='minmax':
		normalized_train_vector = (train_vector - train_vector.min(axis=0))/ (train_vector.max(axis=0) - train_vector.min(axis=0))
		normalized_test_vector = (test_vector - train_vector.min(axis=0))/ (train_vector.max(axis=0) - train_vector.min(axis=0))
		normalized_train_vector=numpy.nan_to_num(normalized_train_vector)
        	normalized_test_vector=numpy.nan_to_num(normalized_test_vector)	
	if normalization_approach=='l1':
		normalized_train_vector=preprocessing.normalize(train_vector, norm='l1')
		normalized_test_vector=preprocessing.normalize(test_vector, norm='l1')
	if normalization_approach=='l2':
		normalized_train_vector=preprocessing.normalize(train_vector, norm='l2')
		normalized_test_vector=preprocessing.normalize(test_vector, norm='l2')
	if normalization_approach=='max':
		normalized_train_vector=preprocessing.normalize(train_vector, norm='max')
		normalized_test_vector=preprocessing.normalize(test_vector, norm='max')
	if normalization_approach=='none':
		normalized_train_vector=train_vector
		normalized_test_vector=test_vector
	return normalized_train_vector, normalized_test_vector

	
	
		

