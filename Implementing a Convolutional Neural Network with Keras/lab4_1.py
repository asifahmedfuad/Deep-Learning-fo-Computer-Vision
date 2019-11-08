# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 01:00:04 2019

@author: Bithi Barua & Kazi Ahmed Asif Fuad
"""

# A cmplete edition

#!pip install -q tensorflow-gpu==2.0.0-beta1
from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop, SGD
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)


#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def show_one_img(X):
    plt.figure(1)
    plt.imshow(Image.fromarray(X))
    plt.show()
    
def show_imgs(X):
    plt.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            plt.subplot2grid((4,4),(i,j))
            plt.imshow(Image.fromarray(X[k]))
            k = k+1
    # show the plot
    plt.show()
    
def show_wrong_imgs(X):
    plt.figure(1)
    k = 0
    for i in range(0,2):
        for j in range(0,5):
            plt.subplot2grid((2,5),(i,j))
            plt.imshow(Image.fromarray(X[k]))
            k = k+1
    # show the plot
    plt.show()
# Display some Iamge 
show_imgs(x_test[:16])
# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8
print('x_train.shape=', x_train.shape)
print('y_test.shape=', y_test.shape)

#To input our values in our network Conv2D layer, we need to reshape the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 28, 28, 1) where 1 is the number of channels of our images
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255

print('x_train.shape=', x_train.shape)
print('x_test.shape=', x_test.shape)

num_classes = 10

#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# num_classes is computed automatically here
# but it is dangerous if y_test has not all the classes
# It would be better to pass num_classes=np.max(y_train)+1


#Let start our work: creating a convolutional neural network

#####TO COMPLETE
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'

    # Compute confusion matrix
    cm = confusion_matrix((y_true>0.5).argmax(axis=1), (y_pred>0.5).argmax(axis=1))
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized Confusion Matrix")
    else:
        cm=cm
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


####Model 

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
sgd = SGD(lr=0.9, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#history=model.fit(x_train, y_train, validation_split=0.20, batch_size=32, epochs=2)
history=model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=32, epochs=50)
score = model.evaluate(x_test, y_test, batch_size=32)
print("Model Test:",score)

#save to disk
model_json = model.to_json()
with open('model41.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model41.h5') 

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


expected = y_test
predicted = model.predict(x_test)
classes=['0','1','2','3','4','5','6','7','8','9']

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plot_confusion_matrix(expected, predicted, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

predicted=model.predict(x_test)
incorrects = np.nonzero(predicted.argmax(axis=1) != y_test.argmax(axis=1))
wrongPredArray=incorrects[0]
wrongPredArrayTen=wrongPredArray[0:10] #first 10 wrong classification
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print((predicted.argmax(axis=1))[wrongPredArrayTen])
show_wrong_imgs(x_test[wrongPredArrayTen])