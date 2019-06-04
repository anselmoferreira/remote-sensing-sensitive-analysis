import numpy as np
import PIL
from PIL import Image
import random
#This function receives images and labels and convert them to numpy matrices in order to be 
#manipulated by keras models
def convert_image_numpy(file_images_train,file_images_test, labels_images_train, labels_images_test, color_space, cnn_type, size_patch):

    with open(file_images_train) as f:

        images_names = f.readlines()
        images_names = [x.strip() for x in images_names]
        array_images_train=[]
                
	    for line in images_names:

            print('Reading Image ' + line)
			im=PIL.Image.open(line)	
			im=im.resize((size_patch,size_patch))  
            im=np.asarray(im).astype(np.float32)						
			array_images_train.append(im)
			
    array_images_train=np.array(array_images_train)
                         
    with open(file_images_test) as f:

        images_names = f.readlines()
        images_names = [x.strip() for x in images_names]
        array_images_test = []
 
        for line in images_names:
                       
            print('Reading Image ' + line)
		    im=PIL.Image.open(line)	
			im=im.resize((size_patch,size_patch))
			im=np.asarray(im).astype(np.float32)                   	
			array_images_test.append(im)

    array_images_test=np.array(array_images_test)

    with open(labels_images_train) as f:

        classes_images = f.readlines()
        classes_images = [x.strip() for x in classes_images]
        array_classes_train = []
   
        for line in classes_images:
            array_classes_train.append(line)

    array_classes_train=np.array(array_classes_train)

    with open(labels_images_test) as f:

        classes_images = f.readlines()
        classes_images = [x.strip() for x in classes_images]
        array_classes_test= []

        for line in classes_images:
            array_classes_test.append(line)

    array_classes_test=np.array(array_classes_test)
	
    if (cnn_type=='mnist' and color_space=='near_infrared'):
		array_images_train = array_images_train.reshape(array_images_train.shape[0], 28, 28,1)
		array_images_test = array_images_test.reshape(array_images_test.shape[0], 28, 28,1)
                     
    return(array_images_train,array_images_test,array_classes_train, array_classes_test)