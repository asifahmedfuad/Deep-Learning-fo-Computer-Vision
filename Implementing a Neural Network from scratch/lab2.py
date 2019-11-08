import numpy as np

#In this first part, we just prepare our data (mnist) 
#for training and testing

import keras
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T
print(X_train.shape)

m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]


# #Display one image and corresponding label 
#import matplotlib
import matplotlib.pyplot as plt
# i = 3
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()


#Let start our work: creating a neural network
#First, we just use a single neuron. 


#####TO COMPLETE
def computeZ(w,X,b):
    z=np.dot(w.T,X)+b
    return z

def sigmoid(z):
    sgm=1/(1+np.exp(-z))
    return sgm

def lossFunction(a,y):
    m = y.shape[1]
    totalLoss=(-1/float(m))*np.sum((y*np.log(a))+((1-y)*np.log(1-a)),axis=1)
    return totalLoss


def optimizeNetwork(X,w,b,y,epochs,lr):
    for iteration in range(epochs):
        z=computeZ(w,X,b)
        A=sigmoid(z)
        totalLoss=lossFunction(A,y)
        print(totalLoss)
        dw = (1/m)*np.dot(X,(A-y).T)
        db = (1/m)*np.sum(A-y,axis=1)
        w=w-lr*dw
        b=b-lr*db
    return w,b
def optNetvalidation(Xtrain,w,b,ytrain,Xtest,ytest,epochs,lr):
    validationLoss=100
    trainingLoss=np.zeros((epochs,1))
    ValidationLoss=np.zeros((epochs,1))
    for iteration in range(epochs):
        print("Iterating "+ str(iteration) + " out of total " + str(epochs) +" epochs")
        ztrain=computeZ(w,Xtrain,b) #training 
        Atrain=sigmoid(ztrain) #training
        #print(Atrain.shape)
        totalLossTrain=lossFunction(Atrain,ytrain)
        trainingLoss[iteration]=totalLossTrain
        print("Training Loss: ", totalLossTrain)
        #back propagation
        dw = (1/float(m))*np.dot(Xtrain,(Atrain-ytrain).T)
        db = (1/float(m))*np.sum(Atrain-ytrain,axis=1)
        w=w-lr*dw
        b=b-lr*db
        ztest=computeZ(w,Xtest,b) #validation on test data 
        Atest=sigmoid(ztest) #validation on test data
        currentLossTest=lossFunction(Atest,ytest)
        #print(Atest.shape)
        ValidationLoss[iteration]=currentLossTest
        print("Validation Loss: ",currentLossTest)
        if currentLossTest>(1.05*validationLoss):
            print("Validation loss is increasing, stopping the training at ",iteration)
            break
        else:
            validationLoss=currentLossTest
    #plt.plot(range(epochs),trainingLoss,'g-',range(epochs),ValidationLoss,'b-')
    fig, ax = plt.subplots()
    ax.plot(range(epochs), trainingLoss, 'g--', label='Training Loss')
    ax.plot(range(epochs), ValidationLoss, 'b-', label='Validation Loss')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('w')
    plt.show()
    return w,b

def predictClass(X,w,b):
    z=computeZ(w,X,b) #training 
    activation=sigmoid(z)
    pclass=activation>0.5
    return pclass
    
weights=np.random.rand(X_train.shape[0],1)*0.01
#weights=np.random.normal(0,0.01,(X_train.shape[0],1))
bias=0
learning_rate=0.9
epochs=500

w,b=optNetvalidation(X_train,weights,bias,y_train,X_test,y_test,epochs,learning_rate)
#w,b=optimizeNetwork(X_train,weights,bias,y_train,epochs,learning_rate)
#print(b)

pred_class_train=predictClass(X_train,w,b)
pred_class_test=predictClass(X_test,w,b)
print("train accuracy: {} %".format(100 - np.mean(np.abs(pred_class_train - y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(pred_class_test - y_test)) * 100))