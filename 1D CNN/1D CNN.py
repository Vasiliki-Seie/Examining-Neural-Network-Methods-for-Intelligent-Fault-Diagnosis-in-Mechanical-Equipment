#1D adaptive CNN for fault detection in bearing faults.
import numpy        
import keras     
import time
import scipy.io as sio     
from sklearn.metrics import confusion_matrix                                   #import of all needed functions
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
K.common.set_image_dim_ordering('th')

start_time = time.time()                                                       #take time measurement after the import decleration

seed=7
numpy.random.seed(seed)                                                        # fix random seed for reproducibility

train_dataX = sio.loadmat('training_dataX.mat')                                #load X,Y training and testing data
test_dataX = sio.loadmat('testing_dataX.mat')  
train_dataY = sio.loadmat('training_dataY.mat')
test_dataY = sio.loadmat('testing_dataY.mat') 

X_train= train_dataX['training_dataX']                                         
Y_train = train_dataY['training_dataY']
X_test = test_dataX['testing_dataX']
Y_test1 = test_dataY['testing_dataY']

X_train = X_train.reshape(X_train.shape[0],200,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],200,1).astype('float32')

Y_train = np_utils.to_categorical(Y_train)                                     #one hot encode outputs
Y_test = np_utils.to_categorical(Y_test1)
num_classes = Y_test.shape[1]

e=0.98                                                                         #create a callback to monitor the error to avoid overfitting
class myCallback(keras.callbacks.Callback):                                            
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > e):   
            print("\nReached %2.2f%% accuracy in last epoch, so stopping training!!" %(e*100))   
            self.model.stop_training = True
Callback = myCallback()

n_timesteps = X_train.shape[1]                                                 #take the X_train size to determine input_shape
n_features = X_train.shape[2]

def baseline_model():                                                          #building our sequential model
    model = Sequential()
    model.add(Conv1D(60,9,input_shape=(n_timesteps,n_features),activation='tanh',padding='same'))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(40,9,activation='tanh',padding='same'))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(40,9,activation='tanh',padding='same'))
    model.add(Flatten())
    model.add(Dense(20,activation='tanh'))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = baseline_model()                                                       #initialize fitting process
model.fit(X_train, Y_train, validation_data=(X_test,Y_test),epochs=100,batch_size=100,callbacks=[Callback])

scores = model.evaluate(X_test,Y_test,verbose=0)                               #final model evaluation
print('CNN Error: %.2f%%' % (100-scores[1]*100))

end_time = time.time()                                                         #calculate and print time it took for the code to execute
execution_time = end_time - start_time
print('CNN Execution time(in sec): %.2f' % execution_time)

predictions = model.predict_classes(X_test, batch_size=100, verbose=0)         #make rounded predictions for the test data
conf_matrix = confusion_matrix(Y_test1, predictions)                           #constract and print the confusion matrix
print('Cofusion Matrix for the test data:')
print(conf_matrix)