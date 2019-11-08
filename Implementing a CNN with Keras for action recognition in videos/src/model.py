from __future__ import print_function
import keras
print('Keras version : ', keras.__version__)
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Conv3D, MaxPooling3D
from keras.optimizers import RMSprop, SGD
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

#############################################
############## Make the model ###############
#############################################


def make_one_branch_model(temporal_dim, width, height, channels, nb_class):
    #TODO
    #Build the 'one branch' model and compile it.
    #Use the following optimizer
    #sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    # This returns a tensor
    inputs = Input(shape=(temporal_dim, width, height, channels))
    output_1 = Conv3D(30, kernel_size=(3, 3, 3), padding='same', activation='relu')(inputs)
    output_2 = MaxPooling3D()(output_1)
    output_3 = Conv3D(60, kernel_size=(3, 3, 3), padding='same', activation='relu')(output_2)
    output_4 = MaxPooling3D()(output_3)
    output_5 = Conv3D(80, kernel_size=(3, 3, 3), padding='same', activation='relu')(output_4)
    output_6 = MaxPooling3D()(output_5)
    output_7 = Flatten()(output_6) 
    output_8 = Dense(500, activation='relu')(output_7) 
    output_9 = Dense(nb_class, activation='softmax')(output_8) 
    model=Model(input=inputs,output=output_9)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def make_model(temporal_dim, width, height, nb_class):
    #TODO
    #Build the siamese model and compile it.
    #Use the following optimizer
    #sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    rgb_input = Input(shape=(temporal_dim, width, height, 3))
    rgb_output_1 = Conv3D(30, kernel_size=(3, 3, 3), padding='same', activation='relu')(rgb_input)
    rgb_output_2 = MaxPooling3D()(rgb_output_1)
    rgb_output_3 = Conv3D(60, kernel_size=(3, 3, 3), padding='same', activation='relu')(rgb_output_2)
    rgb_output_4 = MaxPooling3D()(rgb_output_3)
    rgb_output_5 = Conv3D(80, kernel_size=(3, 3, 3), padding='same', activation='relu')(rgb_output_4)
    rgb_output_6 = MaxPooling3D()(rgb_output_5)
    rgb_output_7 = Flatten()(rgb_output_6) 
    rgb_output= Dense(500, activation='relu')(rgb_output_7) 
    
    flow_input = Input(shape=(temporal_dim, width, height, 2))
    flow_output_1 = Conv3D(30, kernel_size=(3, 3, 3), padding='same', activation='relu')(flow_input)
    flow_output_2 = MaxPooling3D()(flow_output_1)
    flow_output_3 = Conv3D(60, kernel_size=(3, 3, 3), padding='same', activation='relu')(flow_output_2)
    flow_output_4 = MaxPooling3D()(flow_output_3)
    flow_output_5 = Conv3D(80, kernel_size=(3, 3, 3), padding='same', activation='relu')(flow_output_4)
    flow_output_6 = MaxPooling3D()(flow_output_5)
    flow_output_7 = Flatten()(flow_output_6) 
    flow_output = Dense(500, activation='relu')(flow_output_7)
    
    
    both_feat=keras.layers.concatenate(inputs=([rgb_output,flow_output]), axis=-1)
    final_layer = Dense(nb_class, activation='softmax')(both_feat) 
    model=Model(input=[rgb_input,flow_input],output=final_layer)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model
    




