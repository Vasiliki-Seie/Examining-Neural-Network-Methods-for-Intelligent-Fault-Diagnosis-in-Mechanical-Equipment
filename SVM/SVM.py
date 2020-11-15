#SVM for fault detection in bearing faults.
import scipy.io as sio
import pywt                                                                    #import of all needed functions
import numpy
import time
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix

start_time = time.time()                                                       #take time measurement after the import decleration

train_dataX = sio.loadmat('training_dataX.mat')                                #load X,Y training and testing data
test_dataX = sio.loadmat('testing_dataX.mat')  
train_dataY = sio.loadmat('training_dataY.mat')
test_dataY = sio.loadmat('testing_dataY.mat') 

X_train = train_dataX['training_dataX']                                         
Y_train = train_dataY['training_dataY']
X_test = test_dataX['testing_dataX']
Y_test = test_dataY['testing_dataY']

X_train = pywt.dwt(X_train,'bior1.3')                                          #use a wavelet transform filter on our X data
X_test = pywt.dwt(X_test,'bior1.3')

X_train1 = X_train[0]
X_train2 = X_train[1]
X_train = numpy.hstack((X_train1,X_train2))                                    #reshape the X data after the WT to be usable in the SVM

X_test1 = X_test[0]
X_test2 = X_test[1]
X_test = numpy.hstack((X_test1,X_test2))

classification = svm.SVC(kernel='linear')                                      #use linear SVM to define the model
classification.fit(X_train, Y_train)                                           #fit the model

Y_predict = classification.predict(X_test)                                     
Y_predict = Y_predict.reshape(Y_predict.shape[0],1)                            #make predictions for the test data

print("Accuracy:",metrics.accuracy_score(Y_test, Y_predict))                   #calculate accuracy

end_time = time.time()                                                         #calculate and print time it took for the code to execute
execution_time = end_time - start_time
print('SVM Execution time(in sec): %.2f' % execution_time)

conf_matrix = confusion_matrix(Y_test, Y_predict)                              #constract and print the confusion matrix
print('Cofusion Matrix for the test data:')
print(conf_matrix)

