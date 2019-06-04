import numpy as np
import PIL

def converte_data_numpy(file_images, labels_images):

    with open(file_images) as f:

                images_names = f.readlines()
                images_names = [x.strip() for x in images_names]
                array_imagens=[]
                
	        for line in images_names:

                        print('Reading Image ' + line)
			im=PIL.Image.open(line)
                        im=im.resize((32,32))
			im=np.asarray(im).astype(np.float32)
                        array_imagens.append(im)
                      
    array_imagens=np.array(array_imagens)

    with open(labels_images) as f:

                classes_images = f.readlines()
                classes_images = [x.strip() for x in classes_images]
                array_classes = []
   
                for line in classes_images:
                        array_classes.append(line)

    array_classes=np.array(array_classes)
  
                   
    return(array_imagens, array_classes)
