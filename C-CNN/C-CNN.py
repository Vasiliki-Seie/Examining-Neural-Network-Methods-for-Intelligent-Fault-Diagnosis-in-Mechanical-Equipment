#1D CNN with 5 parallel concurrent architechtures with multi-scale kernels for fault detection in bearing faults.
import numpy             
import time
import scipy.io as sio     
from sklearn.metrics import confusion_matrix   
from keras import Model                
from keras import Input                                                        #import of all needed functions
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
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

n_timesteps = X_train.shape[1]                                                 #take the X_train size to determine input_shape
n_features = X_train.shape[2]

n_filters = 64                                                                 #determine the different sizes we need for our layers to be defined
kernel_size = {}
kernel_size[0] = 5
kernel_size[1] = 25
kernel_size[2] = 50
kernel_size[3] = 100
kernel_size[4] = 125
input_shape = (n_timesteps,n_features)
pool_size = 10
n_paraller_filters = 5

inp = Input(shape=input_shape)                                                 #using a for loop, constract the 5 different CNN architechtures which wil run concurrently
convolutions = []
for k in range(len(kernel_size)):
    convolution1 = Conv1D(n_filters,kernel_size[k],padding='same',activation='relu',input_shape=input_shape)(inp)
    pool1 = MaxPooling1D(pool_size=pool_size)(convolution1)
    convolution2 = Conv1D(n_filters,kernel_size[k],padding='same',activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=pool_size)(convolution2)
    convolutions.append(pool2)

out = Concatenate()(convolutions)                                              #using the concatenate layer, extract the combined output of the 5 parallel strucrure

conv_model = Model(input=inp, output=out)                                      #assign the parallel architecture in a variable

def baseline_model():                                                          #built our sequential model
    model = Sequential()
    model.add(conv_model)                                                      #add the parallel architecture as we would any other layer
    model.add(Flatten())
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
    

model = baseline_model()                                                       #initialize fitting process
model.fit(X_train, Y_train, validation_data=(X_test,Y_test),epochs=10,batch_size=100)

scores = model.evaluate(X_test,Y_test,verbose=0)                               #final model evaluation
print('CNN Error: %.2f%%' % (100-scores[1]*100))

end_time = time.time()                                                         #calculate and print time it took for the code to execute
execution_time = end_time - start_time
print('C-CNN Execution time(in sec): %.2f' % execution_time)

predictions = model.predict_classes(X_test, batch_size=100, verbose=0)         #make rounded predictions for the test data
conf_matrix = confusion_matrix(Y_test1, predictions)                           #constract and print the confusion matrix
print('Cofusion Matrix for the test data:')
print(conf_matrix)
    
    