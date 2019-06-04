This folder contain a list of images we augmented to better train the individual networks. We used this list to train and validate both submodels 1 (using near infrared images) and submodel 2 (false color images). To train the submodel #3 we did not do any augmentation because submodel 3 works on vectors. We simply used SMOTE to balance the vectors numbers in different classes.

We used a technique called smart batches to generate these lists. This approach augments samples from the minority class and put them together with the same number of elements of the majority class in the same batch. We used batch size of 32, so, in each batch, there are 16 positive and 16 negative samples. 

We augmented only train and validation images. Testing images are the same and their list can be found in the no_augmented folder.

The approach of smart batches is more detailed in my paper accepted at IEEE JSTARS. You can find it at https://ieeexplore.ieee.org/document/8726132

The images in this list can be downloaded at http://dx.doi.org/10.21227/H2WD42

You may use these images to train your networks, who knows you find something better? :-) 
