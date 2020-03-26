import matplotlib.pyplot as plt
import random
import pickle
import numpy as np
from os import path
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, CallbackList
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.externals import joblib
import os.path


import pandas as pd
import time
from os import listdir
from shutil import copyfile
import pickle

from CNN_Network import *

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def create_seed():
    return random.randint(1, 1000000)

def load_training_images(training_folder, training_file):

    training_data = []
    gibbon_X = []
    noise_X = []
    with open(training_file) as fp:
        line = fp.readline()

        while line:

            file_name = line.strip()
            print()
            print('-----')
            print ('Reading file: {}'.format(file_name))

            if path.exists(training_folder+'g_'+file_name+'_augmented_img.pkl'):
                print ('okay gibbon augmented', file_name)
                gibbon_X.extend(pickle.load(open(training_folder+'g_'+file_name+'_augmented_img.pkl', "rb" )))

            if path.exists(training_folder+'n_'+file_name+'_augmented_img.pkl'):
                print ('okay non-gibbon augmented', file_name)
                noise_X.extend(pickle.load(open(training_folder+'n_'+file_name+'_augmented_img.pkl', "rb" )))

            # Read next line
            line = fp.readline()


    gibbon_X = np.asarray(gibbon_X)
    noise_X = np.asarray(noise_X)

    print('----------------------------------')
    print ('Gibbon features:', gibbon_X.shape)
    print ('Non-gibbon features',noise_X.shape)
    
    return gibbon_X, noise_X

def prepare_X_and_Y(gibbon_X, noise_X):

    Y_gibbon = np.ones(len(gibbon_X))
    Y_noise = np.zeros(len(noise_X))
    X = np.concatenate([gibbon_X, noise_X])
    del gibbon_X, noise_X
    Y = np.concatenate([Y_gibbon, Y_noise])
    del Y_gibbon, Y_noise
    Y = to_categorical(Y)

    return X, Y

def train_model(number_iterations, X, Y):

    for experiment_id in range(0,number_iterations):

        print('Iteration {} starting...'.format(experiment_id))

        print ('experiment_id: {}'.format(experiment_id))
        
        seed = create_seed()
        
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, 
                                                            random_state=seed, shuffle = True)
        
        # Check shape
        print ('X_train:',X_train.shape)
        print ('Y_train:',Y_train.shape)
        print ()
        print ('X_val:',X_val.shape)
        print ('Y_val:',Y_val.shape)

        # Call backs to save weights
        filepath="../Experiments/weights_{}.hdf5".format(seed)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        
        model = network()
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        
        model.summary()
        
        start = time.time()

        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                  batch_size=8,
                  epochs=50,
                  verbose=2, 
                  callbacks=callbacks_list, 
                  class_weight={0:1.,1:1.1})
        end = time.time()
        
        model.load_weights("../Experiments/weights_{}.hdf5".format(seed))
        
        
        train_acc = accuracy_score(model.predict_classes(X_train), np.argmax(Y_train,1))
        print (train_acc)
        
        val_acc = accuracy_score(model.predict_classes(X_val), np.argmax(Y_val,1))
        print (val_acc)
        
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(np.argmax(Y_val,1), model.predict_classes(X_val))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        class_names=['0','1']

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        plt.show()
        
        TN = cnf_matrix[0][0]
        FP = cnf_matrix[0][1]
        FN = cnf_matrix[1][0]
        TP = cnf_matrix[1][1]

        specificity = TN/(TN+FP)
        sensitivity = TP/(FN+TP)

        FPR = 1 - specificity
        FNR = 1 - sensitivity
        
        performance = []
        performance.append(train_acc)
        performance.append(val_acc)
        performance.append(end-start)

        np.savetxt('../Experiments/train_test_performance_{}.txt'.format(seed), np.asarray(performance), fmt='%f') 
        
        with open('../Experiments/history_{}.txt'.format(seed), 'wb') as file_out:
                pickle.dump(history.history, file_out)

        print('Iteration {} ended...'.format(experiment_id))
        print('Results saved to:')
        print('../Experiments/train_test_performance_{}.txt'.format(seed))
        print('-------------------')
        time.sleep(5)