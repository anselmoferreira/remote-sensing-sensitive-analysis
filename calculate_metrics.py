import numpy as np
from sklearn.metrics import f1_score

#A code to calculate the confusion matrix plus the other metrics
def calculate_metrics(predict,groundtruth,set_testing,classifier):
    
    np.set_printoptions(suppress=True)
    confusion=np.zeros((2,2))
    groundtruth=groundtruth.astype(int)
	
	predict=predict.astype(int)
	    for row,_ in np.ndenumerate(predict):
		    x=groundtruth[row]
		    y=predict[row]
    		confusion[x,y]=confusion[x,y]+1
	print(confusion)

	total_matrix=np.sum(confusion)
	total_diagonal=np.trace(confusion)
	accuracy=(total_diagonal/total_matrix)*100

	TPR=100*(confusion[1,1]/(confusion[1,1]+confusion[1,0]))
	FNR=100-TPR
	TNR=100*(confusion[0,0]/(confusion[0,0]+confusion[0,1]))
	FPR=100-TNR

	balanced_accuracy=((TPR+TNR)/2)
        
	print(confusion)	
	print(balanced_accuracy)

	fmeasure=f1_score(groundtruth, predict)  
	final_result=np.hstack((fmeasure, balanced_accuracy, TPR, TNR, FPR, FNR))

	f_handle = open('../results/'+classifier+'-result.txt','a')
	np.savetxt(f_handle,final_result[None,:], delimiter=',', fmt='%5.2f')
	f_handle.close()

    print("Normal Accuracy =" + accuracy.astype('|S5')+ "%")
    print("Balanced Accuracy=" + balanced_accuracy.astype('|S5')+ "%")
	print('TPR=' + TPR.astype('|S5')+ "%")
    print('FNR=' + FNR.astype('|S5')+ "%")
    print('TNR=' + TNR.astype('|S5')+ "%")
    print('FPR=' + FPR.astype('|S5')+ "%")

	return(balanced_accuracy,fmeasure)
