from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from keras import backend as K
import load_input_data as id
from sklearn.svm import SVC
import feature_vectors_generator as fvg
from sklearn import svm
import calculate_metrics as cs
import keras
from scipy import stats
from sklearn import preprocessing
import normalize_vectors as nv
import numpy as np

#This is the code you should run to reproduce one version of our experiments
#Here we apply two CNNs on different remote sensing data, merge, normalize and use the vectors for training and testing an SVM classifier
#Input parameters are:
#set_training: the bag you are using for training. 1 or 2
#set_testing: the bag you are using for testing, 1 or 2 
#normalization: Normalization used in the concatenated vectors. We used min-max as in the paper
#NOTE: you can try to train the networks in your server, but you will hardly get the same result from us as the weights in CNNs
#are always started "semi-randomly". But it is a good starting point for you to learn and play with your own CNN in this problem
#NOTE 2: the original approach uses another network (submodel #3) in the concatenated normalized vectors. However, I forgot to 
#save the model and weights for this network in both validation steps. The code contains a variation, which uses the vectors as 
#input for SVMs. The accuracy difference is not too much.
#You can train the submodel #3 in these vectors by yourself (the submodel #3 training code is on train_the_models folder), 
#save the weigths and model definition and adapt this code. 
#Dont hesitate to contact me if you have further doubts
def main(set_training, set_testing, normalization):

	#Load submodel #1: Lenet trained on near infrared images 
	json_file = open('models/submodel1/'+set_training+'-'+set_testing+'-0.01.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	mnist_loaded_model = model_from_json(loaded_model_json)
	#Load submodel #1 weights
	mnist_loaded_model.load_weights('models/submodel1/'+set_training+'-'+set_testing+'-0.01.h5')
	inputs = [K.learning_phase()] + mnist_loaded_model.inputs
    #I get the last but one layer in order to get the vectors
	_mnist_f = K.function(inputs, [mnist_loaded_model.layers[len(mnist_loaded_model.layers)-2].output])
	print("Loaded MODEL #1 from disk")

 	#Load Submodel #2: Densenet trained on false color images
	json_file = open('models/submodel2/'+set_training+'-'+set_testing+'-0.01.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	densenet2_loaded_model = model_from_json(loaded_model_json)
	#Load submodel #2 weights
	densenet2_loaded_model.load_weights('models/submodel2/'+set_training+'-'+set_testing+'-0.01.h5')
	inputs = [K.learning_phase()] + densenet2_loaded_model.inputs
    #I get the last but one layer in order to get the vectors
	_densenet_f = K.function(inputs, [densenet2_loaded_model.layers[len(densenet2_loaded_model.layers)-2].output])
	print("Loaded MODEL #2 from disk")

	#Load Submodel #3: Shallow network on normalized fused features
	json_file = open('models/submodel3/'+set_training+'-'+set_testing+'-0.01.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	submodel3_loaded_model = model_from_json(loaded_model_json)
	#Load submodel #2 weights
	submodel3_loaded_model.load_weights('models/submodel3/'+set_training+'-'+set_testing+'-0.01.h5')
	inputs = [K.learning_phase()] + submodel3_loaded_model.inputs
	print("Loaded MODEL #3 from disk")
	
	#Now, we load data to apply in this models, in order to generate feature vectors
	# First of all, lets load 28x28 near infrared images
	x_train_mnist_ni, x_test_mnist_ni, y_train_mnist_ni, y_test_mnist_ni = id.convert_image_numpy('fold-data/near_infrared/bag'+set_training+'-images.txt', 'fold-data/near_infrared/bag'+set_testing+'-images.txt', 'fold-data/near_infrared/bag'+set_training+'-labels.txt', 'fold-data/near_infrared/bag'+set_testing+'-labels.txt', 'near_infrared', 'mnist', 28)
	# Lets normalize these data
	x_train_mnist_ni = x_train_mnist_ni.astype('float32')
	x_test_mnist_ni = x_test_mnist_ni.astype('float32')
	x_train_mnist_ni /= 255
	x_test_mnist_ni /= 255

	#Now we load  32x32 False color images from the dataset
	x_train_densenet_fc, x_test_densenet_fc, y_train_densenet_fc, y_test_densenet_fc=id.convert_image_numpy('fold-data/false_color/bag'+set_training+'-images.txt', 'fold-data/false_color/bag'+set_testing+'-images.txt', 'fold-data/false_color/bag'+set_training+'-labels.txt', 'fold-data/false_color/bag'+set_testing+'-labels.txt', 'false_color', 'densenet', 32)
	#normalize data again
	x_train_densenet_fc = x_train_densenet_fc.astype('float32')
	x_test_densenet_fc = x_test_densenet_fc.astype('float32')
	x_train_densenet_fc /= 255
	x_test_densenet_fc /= 255

	#############################################################################################################################################################################################
	print("Generating Feature Vectors")
	#Lets apply the two models loaded in the images in order to generate feature vectors 
	features_train_mnist1, features_test_mnist1=fvg.feature_vectors_generator(x_train_mnist_ni,x_test_mnist_ni,_mnist_f)
	features_train_densenet2, features_test_densenet2=fvg.feature_vectors_generator(x_train_densenet_fc,x_test_densenet_fc,_densenet_f)
	#Reshaping the feature vectors
	y_train_mnist_ni=y_train_mnist_ni.reshape(y_train_mnist_ni.shape[0],1)	
	y_test_mnist_ni=y_test_mnist_ni.reshape(y_test_mnist_ni.shape[0],1)	
	y_train_densenet_fc=y_train_densenet_fc.reshape(y_train_densenet_fc.shape[0],1)	
	y_test_densenet_fc=y_test_densenet_fc.reshape(y_test_densenet_fc.shape[0],1)	
    ##############################################################################################################################################################################################
	print("Concatenating and normalizing")
	#Concatenating feature vectors from both CNNs
	train_data=numpy.column_stack((features_train_densenet2, features_train_mnist1))
	test_data=numpy.column_stack((features_test_densenet2, features_test_mnist1))
	#Normalizing them with minmax normalization, as explained in the paper
	final_train_vector,final_test_vector=nv.normalize(train_data, test_data, normalization)
	print("Saving Feature Vectors")
	#Just saving them, in order to use them later in another classifier, such as the submodel 3
	#You can study these features later, reducing dimensionality, selecting best features with 
	#Random Forests, etc.
	f_handle = open('feature_vectors/fold1-bag'+set_training+'.txt','w')
	np.savetxt(f_handle, final_train_vector, delimiter=',', fmt='%s') 	
	f_handle.close()
	f_handle = open('feature_vectors/fold1-bag'+set_testing+'.txt','w')
	np.savetxt(f_handle, final_test_vector, delimiter=',', fmt='%s') 	
	f_handle.close()

	#Below I was using the features from the different models
	#to train and test an SVM classifier
	#I did this in order to make a fair comparison with the state of the art, 
	#which also used SVMs.
	print("Training and Classifying- Proposed Model #1")
	linear_svcmodel1 = svm.SVC(kernel='linear', class_weight='balanced')
	linear_svcmodel1.fit(features_train_mnist1, y_train_mnist_ni)
	predictmodel1=linear_svcmodel1.predict(features_test_mnist1)
	
	print("Training and Classifying- Proposed Model #2")
	linear_svcmodel2 = svm.SVC(kernel='linear', class_weight='balanced')
	linear_svcmodel2.fit(features_train_densenet2, y_train_densenet_fc)
	predictmodel2=linear_svcmodel2.predict(features_test_densenet2)
	
	print("Training and Classifying- Proposed Fusion")
	X_train, Y_train, groundtruth_train= load_vector.loadvector_train(train_vector_file, "single_smote")
	linear_svc.fit(final_train_vector, y_train_densenet_fc)
	predict=linear_svc.predict(final_test_vector)
	
	print("Calculating metrics- Model #1")
	final_accuracy, fmeasure=cs.calculate_statistics(predictmodel1, y_test_mnist_ni, set_testing, 'SVM')
	print("Proposed Model #1 f-measure:"+str(fmeasure))
	print("Proposed Model #1 Accuracy:"+str(final_accuracy))
	
	print("Calculating metrics- Model #2")
	final_accuracy, fmeasure=cs.calculate_statistics(predictmodel2, y_test_densenet_fc, set_testing, 'SVM')
	print("Proposed Model #2 f-measure:"+str(fmeasure))
	print("Proposed Model #2 Accuracy:"+str(final_accuracy))
	
	#I am not using SMOTE nor another CNN as you can see
	#This happens because I lost the submodel's 3 weights for both bags :-(
	#You can try SMOTE here, or you can try training submodel 3 from the folder train_the_models
	#and adapt the code in order to make it classifying the input features
	#The difference will not be too much (0.5/1% in accuracy)
	#If you need some help on it drop me a message at anselmo.ferreira@gmail.com	
	print("Calculating metrics- Proposed Model")
	final_accuracy, fmeasure=cs.calculate_statistics(predict, y_test_densenet_fc, set_testing, 'SVM')
	print("Proposed Fusion Method f-measure:"+str(fmeasure))
	print("Proposed Fusion Method Accuracy:"+str(final_accuracy))
	print("Press any key to continue...")
	raw_input()