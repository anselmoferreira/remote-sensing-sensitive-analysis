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
import calculate_statistic as cs
import keras
from scipy import stats
from sklearn import preprocessing
import normalize_vectors as nv
import numpy as np

def main(set_training, set_testing, normalization):

	#MODEL #1 LENET ON NEAR INFRARED IMAGE
	json_file = open('models/lenet-nir/'+set_training+'-'+set_testing+'-0.01.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	mnist_loaded_model = model_from_json(loaded_model_json)
	mnist_loaded_model.load_weights('models/lenet-nir/'+set_training+'-'+set_testing+'-0.01.h5')
	inputs = [K.learning_phase()] + mnist_loaded_model.inputs
    _mnist_f = K.function(inputs, [mnist_loaded_model.layers[len(mnist_loaded_model.layers)-2].output])
	print("Loaded MODEL #1 from disk")

 	#MODEL #2: DENSENET ON FALSE COLOR IMAGE
	json_file = open('models/densenet-fc/'+set_training+'-'+set_testing+'-0.01.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	densenet2_loaded_model = model_from_json(loaded_model_json)
	densenet2_loaded_model.load_weights('models/densenet-fc/'+set_training+'-'+set_testing+'-0.01.h5')
	inputs = [K.learning_phase()] + densenet2_loaded_model.inputs
    _densenet_f = K.function(inputs, [densenet2_loaded_model.layers[len(densenet2_loaded_model.layers)-2].output])
	print("Loaded MODEL #2 from disk")
	
	#Now, we load data (28x28 NEAR INFRARED)
	x_train_mnist_ni, x_test_mnist_ni, y_train_mnist_ni, y_test_mnist_ni = id.convert_image_numpy('fold-data/near_infrared/bag'+set_training+'-images.txt', 'fold-data/near_infrared/bag'+set_testing+'-images.txt', 'fold-data/near_infrared/bag'+set_training+'-labels.txt', 'fold-data/near_infrared/bag'+set_testing+'-labels.txt', 'infravermelho_proximo', 'mnist', 28)
	x_train_mnist_ni = x_train_mnist_ni.astype('float32')
	x_test_mnist_ni = x_test_mnist_ni.astype('float32')
	x_train_mnist_ni /= 255
	x_test_mnist_ni /= 255

	#Now we load data (32x32 FALSE COLOR)
	x_train_densenet_fc, x_test_densenet_fc, y_train_densenet_fc, y_test_densenet_fc=id.convert_image_numpy('fold-data/false_color/bag'+set_training+'-images.txt', 'fold-data/false_color/bag'+set_testing+'-images.txt', 'fold-data/false_color/bag'+set_training+'-labels.txt', 'fold-data/false_color/bag'+set_testing+'-labels.txt', 'falsa_cor', 'densenet', 32)
	x_train_densenet_fc = x_train_densenet_fc.astype('float32')
	x_test_densenet_fc = x_test_densenet_fc.astype('float32')
	x_train_densenet_fc /= 255
	x_test_densenet_fc /= 255

	############################################################################################################################################################################################3
	print("Generating Feature Vectors")
	features_train_mnist1, features_test_mnist1=fvg.feature_vectors_generator(x_train_mnist_ni,x_test_mnist_ni,_mnist_f)
	features_train_densenet2, features_test_densenet2=fvg.feature_vectors_generator(x_train_densenet_fc,x_test_densenet_fc,_densenet_f)
	
	y_train_mnist_ni=y_train_mnist_ni.reshape(y_train_mnist_ni.shape[0],1)	
	y_test_mnist_ni=y_test_mnist_ni.reshape(y_test_mnist_ni.shape[0],1)	
	y_train_densenet_fc=y_train_densenet_fc.reshape(y_train_densenet_fc.shape[0],1)	
	y_test_densenet_fc=y_test_densenet_fc.reshape(y_test_densenet_fc.shape[0],1)	

	print("Concatenating and normalizing")
	train_data=numpy.column_stack((features_train_densenet2, features_train_mnist1))
	test_data=numpy.column_stack((features_test_densenet2, features_test_mnist1))
	final_train_vector,final_test_vector=nv.normalize(train_data, test_data, normalization)
	print("Saving Feature Vectors")
	f_handle = open('feature_vectors/fold1-bag'+set_training+'.txt','w')
	np.savetxt(f_handle, final_train_vector, delimiter=',', fmt='%s') 	
	f_handle.close()
	f_handle = open('feature_vectors/fold1-bag'+set_testing+'.txt','w')
	np.savetxt(f_handle, final_test_vector, delimiter=',', fmt='%s') 	
	f_handle.close()

	print("Training and Classifying- Proposed Model #1")
	linear_svcmodel1 = svm.SVC(kernel='linear', class_weight='balanced')
	linear_svcmodel1.fit(features_train_mnist1, y_train_mnist_ni)
	predictmodel1=linear_svcmodel1.predict(features_test_mnist1)

	print("Training and Classifying- Proposed Model #2")
	linear_svcmodel2 = svm.SVC(kernel='linear', class_weight='balanced')
	linear_svcmodel2.fit(features_train_densenet2, y_train_densenet_fc)
	predictmodel2=linear_svcmodel2.predict(features_test_densenet2)

	print("Training and Classifying- Proposed Fusion")
	linear_svc = svm.SVC(kernel='linear', class_weight='balanced')
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

	print("Calculating metrics- Proposed Model")
	final_accuracy, fmeasure=cs.calculate_statistics(predict, y_test_densenet_fc, set_testing, 'SVM')
	print("Proposed Fusion Method f-measure:"+str(fmeasure))
	print("Proposed Fusion Method Accuracy:"+str(final_accuracy))
	print("Press any key to continue...")
	raw_input()