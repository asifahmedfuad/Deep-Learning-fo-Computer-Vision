# -*- coding: utf-8 -*-
from __future__ import print_function
import time, random, datetime, gc
from src.functions import *
from src.model import *
from src.data import *


###########################################################
######################### Script ##########################
###########################################################
if __name__ == "__main__":
    # extract_RGB()
    # stat_dataset()
    # compute_flow()

    #data_type='Flow'
    data_type='RGB'

    name_script = 'keras_%s_%s' % (data_type, datetime.datetime.now().strftime("%d-%m-%Y_%H-%M"))
    flow_normalization = get_flow_normalization(visualization = 1) #changing the visualization to 0
    class_index, train_list, test_list = define_list(train_file = '/net/ens/DeepLearning/lab5/Data_TP/list/trainlist.txt',
                                                     test_file = '/net/ens/DeepLearning/lab5/Data_TP/list/testlist.txt',
                                                     class_file ='/net/ens/DeepLearning/lab5/Data_TP/list/classInd.txt')

    width = 100; height = 100; temporal_dim = 100; nb_class = len(class_index); batch_size = 10; nb_epoch = 30 #30
    random.seed(1)

    train_generator = My_Data_Sequence_one_branch(train_list, flow_normalization, class_index, batch_size, data_type=data_type, augmentation=True)
    test_generator = My_Data_Sequence_one_branch(test_list[::10], flow_normalization, class_index, batch_size, data_type=data_type, shuffle=False)
    channels = 3
    weights_file='/net/ens/DeepLearning/lab5/Data_TP/models/RGB_model.hdf5'
    if data_type == 'Flow':
        channels = 2
        weights_file='/net/ens/DeepLearning/lab5/Data_TP/models/Flow_model.hdf5'

    #TODO
    #1) create 'one branch' model
    #2) load weights
    model=make_one_branch_model(temporal_dim, width, height, channels, nb_class)
    model.load_weights(weights_file)


    log_dir = './TensorBoard/%s' % name_script; 
    remove_file(log_dir); 

    # Callbacks
    tensorboard_call = keras.callbacks.TensorBoard(log_dir = log_dir, batch_size=batch_size, write_images=False)
    log_callback = keras.callbacks.CSVLogger(os.path.join('logs', '%s.log' % name_script), separator=',', append=False)
    checkpoint_weigths = keras.callbacks.ModelCheckpoint('models_saved/weigths_%s_epoch_{epoch:03d}-valloss_{val_loss:.3f}.hdf5' % name_script, save_weights_only=True) #first


    print('Training :'); tic = time.time()
    #TODO
    #3) train model
    #   using: callbacks = [tensorboard_call, checkpoint_weigths, log_callback]
    model.fit_generator(train_generator,epochs=nb_epoch,validation_data=test_generator,callbacks = [tensorboard_call, checkpoint_weigths, log_callback])
    score = model.evaluate_generator(test_generator,verbose=0)
    print('\nTest result: %.3f loss: %.3f' % (score[1]*100,score[0]))
    print('Training time : %.3g s' % (time.time() - tic))
    


    
