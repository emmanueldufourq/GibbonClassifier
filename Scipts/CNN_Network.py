from keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D
from keras.models import Sequential
import keras
import tensorflow as tf

def network():
    model = Sequential()
    
    model.add(keras.layers.Conv2D(8, 4, input_shape = (128,188,1), padding='same', activation='relu'))
    model.add(keras.layers.Dropout(0.40))
    model.add(keras.layers.MaxPool2D(4,))

    model.add(keras.layers.Conv2D(16, 4, padding='same', activation='relu'))
    model.add(keras.layers.Dropout(0.40))
    model.add(keras.layers.MaxPool2D(4,))
    
    model.add(keras.layers.Conv2D(16, 4, padding='same', activation='relu'))
    model.add(keras.layers.Dropout(0.40))
    model.add(keras.layers.MaxPool2D(4,))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.40))

    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))
    
    return model