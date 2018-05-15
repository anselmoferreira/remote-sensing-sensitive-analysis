import numpy
import main

def feature_vectors_generator(x_train,x_test, _convout1_f):

        print('Generating Training Feature Vectors...')
  
        batch_size=100
        index=0
        if x_train.shape[0]%batch_size==0:
                max_iterations=x_train.shape[0]/batch_size
        else:
                max_iterations=(x_train.shape[0]/batch_size)+1

        
        for i in xrange(0, max_iterations):
                
                if(i==0):
                        features = _convout1_f([0] + [x_train[index:batch_size]])
                        index=index+batch_size
                        features = numpy.squeeze(features)
			features_train = features

                else:
                         if(i==max_iterations-1):
                                features = _convout1_f([0] + [x_train[index:x_train.shape[0],:]])
                                features = numpy.squeeze(features)
                                features_train = numpy.append(features_train,features, axis=0)

                         else:
                                features =_convout1_f([0] + [x_train[index:index+batch_size,:]])
                                index=index+batch_size
                                features = numpy.squeeze(features)
                                features_train=numpy.append(features_train,features, axis=0)
                                                                                                
	print('Generating Testing Feature Vectors...')
	
	batch_size=100
    index=0
    
    if x_test.shape[0]%batch_size==0:
            max_iterations=x_test.shape[0]/batch_size
    else:
            max_iterations=(x_test.shape[0]/batch_size)+1


    for i in xrange(0, max_iterations):

        if(i==0):
            features = _convout1_f([0] + [x_test[index:batch_size]])
            index=index+batch_size
            features = numpy.squeeze(features)
            features_test = features

        else:
             if(i==max_iterations-1):
                features = _convout1_f([0] + [x_test[index:x_test.shape[0],:]])
                features = numpy.squeeze(features)
                features_test = numpy.append(features_test,features, axis=0)
                
             else:
                features =_convout1_f([0] + [x_test[index:index+batch_size,:]])
                index=index+batch_size
                features = numpy.squeeze(features)
                features_test=numpy.append(features_test,features, axis=0)

	return(features_train, features_test)