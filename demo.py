import main 
import tensorflow as tf

#Call the experiment 1, qhich will use bag1 images used to train the model
#and bag2 images to test the model
#Fused features will be normalized by minmax before being fed to the submodel#3
main.main("1","2", 'minmax')
#Here, the same happens. But bag1 images are used now to test and bag2 images are used to train
#Again, minmax is applied in the fused features
main.main("2","1", 'minmax')
