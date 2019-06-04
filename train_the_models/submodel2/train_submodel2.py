from __future__ import print_function

import os.path
import densenet
import numpy as np
import sklearn.metrics as metrics

#use this if you have a gpu
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import load_input_data as id
from sklearn import svm
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model

#If you want to generate feature vectors later
from keras import backend as K


batch_size = 32
nb_classes = 2
nb_epoch = 300
img_rows, img_cols = 32, 32
img_channels = 3
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 16
dropout_rate = 0.0
img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)

#This is the code you should run to train from the scratch the submodel #1 CNN
#Input parameters are:
#train_smart_batch_data: text file containing the training data built via smart batches (check the gitHub repo for more details)
#validation_smart_batch_data: same for validation 
#train_smart_batch_labels: same for training labels
#validation_smart_batch_labels: same for validation labels
#image_type: the kind of image you are using (e.g., near infrared or false color) 
#set_training: the bag you are using for training. 1 or 2
#set_testing: the bag you are using for testing 
#learning_rate=0.01 to train the network
#NOTE: you can try to train the network in your server, but you will hardly get the same result from us as the weights in CNNs
#are always started "semi-randomly". But it is a good starting point for you to learn and play with your own CNN

def train_network(train_smart_batch_data, validation_smart_batch_data, train_smart_batch_labels, validation_smart_batch_labels, image_type, set_training, set_testing, learning_rate=0.01):

	#Load training smart batches data (about 320000 false color images)
    #Converts the input images (x) and labels (y) to the numpy format, which is what keras understands
	x_train_sb, y_train_sb = id.convert_data_numpy(train_smart_batch_data, train_smart_batch_labels)
	#Load validation smart batches data (about 32000 false color images)
    #Converts the input images (x) and labels (y) to the numpy format, which is what keras understands
	x_validation_sb, y_validation_sb = id.convert_data_numpy(validation_smart_batch_data, validation_smart_batch_labels)
	#Converts labels (y) again to the categorical format, a way that keras understands
	y_train_sb = np_utils.to_categorical(y_train_sb, nb_classes)
	y_validation_sb = np_utils.to_categorical(y_validation_sb, nb_classes)
	#normalizes data
	x_train_sb = x_train_sb.astype('float32')
	x_validation_sb=x_validation_sb.astype('float32')
	x_train_sb /= 255.
	x_validation_sb /= 255.
	#Load a base model. Our model is based on the Densenet model. 
	#But trained from scratch and with other differences already pointed out in the paper 
	base_model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block, include_top=False, growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate)
	base_model.summary()
	#Add last layer to our base model based in our problem
	last = base_model.output
	x = Dense(64, activation='relu')(last)
	preds = Dense(2, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=preds)
	model.summary()
	
	#For submodel #2, we used ADADELTA 
	adadelta=keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=["accuracy"])
	print("Finished compiling")

	#To save weights
	out_dir="weights/"
	weights_file="weights/"+image_type+set_training+"-"+set_testing+".h5"
	
	#An approach to learning rate reducing through training
	lr_reducer= ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
	#An approach to stop training before the whole epochs are processed
	early_stopper=EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
	#Policy to save weights
	model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True, save_weights_only=True,mode='auto')
	#Callbacks
	callbacks=[lr_reducer,early_stopper,model_checkpoint]
	#Train the network
	model.fit(x_train_sb, y_train_sb, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks, verbose=2, validation_data=(x_validation_sb, y_validation_sb))

	#Save the model
	model_json = model.to_json()
	with open(set_training+"-"+set_testing+"-"+str(learning_rate)+".json", "w") as json_file:
    	json_file.write(model_json)

	#You can use this code if you want to generate feature vectors
	#If you want to generate features to testing images too, you need to load testing images, as 
	#you did with training and validation data.
	#Dont forget that, if you want to save the feature vector files, you must use the same libsvm format
	#In other words, its a CSV file where the first value is the label, and the rest are the features
	#inputs = [K.learning_phase()] + model.inputs
    #_convout1_f = K.function(inputs, [model.layers[len(model.layers)-2].output])
 	#features_train=fvg.feature_vectors_generator(x_train_sb,_convout1_f)
    #features_teste=fvg.feature_vectors_generator(x_test_sb,_convout1_f)