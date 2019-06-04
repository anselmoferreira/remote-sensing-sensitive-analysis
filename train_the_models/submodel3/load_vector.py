from keras.utils import np_utils
import os
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE

#just to load the vectors from files
#the files are CSV files with the label (y) in the first value and (x) in the remaining values. Everything separated by commas
def loadvector(file_name):
	data = np.genfromtxt(file_name, delimiter=',')
	y=data[:,0]
	x=data[:,1:data.shape[1]]
	x=x.reshape(x.shape[0],x.shape[1],1)
	return x,np_utils.to_categorical(y, 2), y

def loadvector_train(file_name, type):

	data = np.genfromtxt(file_name, delimiter=',')
        y=data[:,0]
        x=data[:,1:data.shape[1]]

	if (type=="single_smote"):
        	sm = SMOTE(random_state=42)
        	x, y = sm.fit_sample(x, y)
	

	x=x.reshape(x.shape[0],x.shape[1],1)
        return x,np_utils.to_categorical(y, 2), y




