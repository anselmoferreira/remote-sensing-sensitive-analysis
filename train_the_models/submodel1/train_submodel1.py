from __future__ import print_function
#If you want to use a gpu, uncomment this
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import theano
#If you want to generate feature vectors later
from keras import backend as K

batch_size = 32
num_classes = 2
epochs=300
data_augmentation = False
import load_input_data as id
import numpy

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
	#input size of submodel #1 (28x28)
	img_rows, img_cols = 28, 28
	#Load smart batch training data (about 320000 near infrared images)
	#Converts the input images (x) and labels (y) to the numpy format, which is what keras understands
	x_train_sb,  y_train_sb = id.convert_data_numpy(train_smart_batch_data, train_smart_batch_labels)
	#Load smart batch validation data (about 32000 near infrared images)
	#Converts the input images (x) and labels (y) to the numpy format, which is what keras understands
	x_validation_sb,  y_validation_sb = id.convert_data_numpy(validation_smart_batch_data, validation_smart_batch_labels)
	#Converts labels (y) again to the categorical format, a way that keras understands
	y_train_sb = np_utils.to_categorical(y_train_sb, num_classes)
	y_validation_sb = np_utils.to_categorical(y_validation_sb, num_classes)
	#Reshaping data
	x_train_sb = x_train_sb.reshape(x_train_sb.shape[0], img_rows, img_cols, 1)
	x_validation_sb = x_validation_sb.reshape(x_validation_sb.shape[0], img_rows, img_cols, 1)
	#Normalizes data
	x_train_sb = x_train_sb.astype('float32')
	x_validation_sb=x_validation_sb.astype('float32')
	x_train_sb /= 255.
	x_validation_sb /= 255.
	#Define the input shape (28x28 patches)
    input_shape = (img_rows, img_cols, 1)
	#This is the CNN used, as reported in the paper
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer='glorot_normal'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_normal'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128))
	convout1=Activation('relu')
    model.add(convout1)
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax')) 

    #We used adamax as optimizer, as we discussed in the paper
	adamax=keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#Compile the model
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=nadam,metrics=['accuracy'])
	#where to save the weights
	out_dir="weights/"
	weights_file="weights/"+image_type+set_training+"-"+set_testing+".h5"
	#An approach to learning rate reducing through training
	lr_reducer= ReduceLROnPlateau(monitor='val_loss', factor=numpy.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
	#An approach to stop training before the whole epochs are processed
	early_stopper=EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
    #Policy to save weights
	model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True, save_weights_only=True,mode='auto')
	#callbacks
	callbacks=[lr_reducer,early_stopper,model_checkpoint]
    #Train the network
	model.fit(x_train_sb, y_train_sb, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=2, validation_data=(x_validation_sb, y_validation_sb))
	
	#Save model
	model_json = model.to_json()
	with open(set_training+"-"+set_testing+"-"+str(learning_rate)+".json", "w") as json_file:
    	json_file.write(model_json)

	#You can use this code if you want to generate feature vectors
	#If you want to generate features to testing images too, you need to load testing images, as 
	#you did with training and validation data.
	#Dont forget that, if you want to save the feature vector files, you must use the same libsvm format
	#In other words, its a CSV file where the first value is the label, and the rest are the features.
	#inputs = [K.learning_phase()] + model.inputs
    #_convout1_f = K.function(inputs, [model.layers[len(model.layers)-2].output])
 	#features_train=fvg.feature_vectors_generator(x_train_sb,_convout1_f)
    #features_teste=fvg.feature_vectors_generator(x_test_sb,_convout1_f)