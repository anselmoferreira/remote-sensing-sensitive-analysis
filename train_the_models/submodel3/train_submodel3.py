import load_vector
from keras import backend as K
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten
from keras.models import Model
import numpy as np
from keras import optimizers
from keras.layers import Conv1D
from keras.callbacks import LearningRateScheduler, CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold='nan')

#use this if you want to use gpu
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

#This is the code you should run to train from the scratch the submodel #3 CNN
#This receives as input the normalized fused vectors from submodels #1 and #2
#Input parameters are:
#train_vector_file: Libsvm style file of feature vectors and labels for training data
#optimizer_name: algorithm to update weights. We used RMSPROP
#NOTE: the files must contain feature vectors normalized with min-max
#There is a file called normalize_vectors.py in the root directory that can help you doing that
	
def train_network(train_vector_file, optimizer_name):
	#Some parameters of the CNN
	batch_size = 32
	nb_classes = 2
	nb_epoch = 100
	
	#This is submodel #3 CNN structure, as presented in the paper
	#This CNN will receive the fused min-max normalized vectors from submodels #1 and #2 
	input = Input(shape=(192,1))
	bn=BatchNormalization()(input)
	conv1=Conv1D(1,32, strides=1, padding='valid', activation='relu')(bn)
	fully_connected=Dense(160, activation='relu')(conv1)
	dropout=Dropout(0.5)(fully_connected)
	flatten=Flatten()(dropout)
	predictions = Dense(2, activation='softmax')(flatten)
	model = Model(inputs=input, outputs=predictions)
	model.compile(loss='categorical_crossentropy',optimizer=optimizer_name, metrics=['accuracy'])
	
	#load the vectors of training data
	#Here, we use the original vectors created from original images (not augmented)
	#Of course, you could use the augmented images, but this could be unfair with the state of the art as
	#you needed to do the same for them. We also did not do that because we did a very huge comparison with dozens of approaches, 
	#and training dozens of SVMs on more than 300k vectors would take months to finish the experiments
	#So, we used the original images only to capture the vectors from submodels #1 and #2, 
	#which must be saved on train_vector_file (and test_vector_file if you want to test the model)
	#As the vectors have unbalanced nature in the distribution of the classes, we perform SMOTE to balance them up when training
	#NOTE: the saved files must contain feature vectors normalized with min-max
	#There is a file called normalize_vectors.py in the root directory that can help you doing that
	X_train, Y_train, groundtruth_train= load_vector.loadvector_train(train_vector_file, "single_smote")
	
	#Split the training data on 70% for training and 30% for validation 
	x_train, x_validation, y_train, y_validation = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)      
          
	lr_reducer= ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=15)
    csv_logger = CSVLogger('vector_cnn.csv')
    model_names = 'weights/' +optimizer_name+  '.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
	
	#Train the network
	model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(x_validation, y_validation), shuffle=True, callbacks=[lr_reducer, early_stopper, csv_logger,model_checkpoint])
	
	#Use this code to predict testing data. You must load them as you did with training data
	#prediction=model.predict(x_test, batch_size=32, verbose=0)
	#predict=np.argmax(prediction, axis=1)
