from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D
from tensorflow.keras.models import Sequential

def network_2D(conv_layers, fc_layers, conv_filters, dropout_rate, 
               conv_kernel,max_pooling_size,fc_units,epochs,batch_size):
    
    model = Sequential()
    model.add(Conv2D(filters = conv_filters, kernel_size = conv_kernel, input_shape = (128,188,1)
                     ,activation = 'relu'))
    model.add(Dropout(rate = dropout_rate))
    model.add(MaxPool2D(pool_size = max_pooling_size))
    
    for i in range(conv_layers):
        model.add(Conv2D(filters = conv_filters, kernel_size = conv_kernel, activation = 'relu'))
        model.add(Dropout(rate=dropout_rate))
        model.add(MaxPool2D(pool_size=max_pooling_size))
        
    model.add(Flatten())
    for i in range(fc_layers):
        model.add(Dense(units = fc_units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
        
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    return model

def network():

    conv_layers = 1
    fc_layers = 1
    max_pooling_size = 4
    dropout_rate = 0.4
    conv_filters = 8
    conv_kernel = 16
    fc_units = 32
    epochs = 40
    batch_size = 8
    model = network_2D(conv_layers, fc_layers, conv_filters, dropout_rate, 
               conv_kernel,max_pooling_size,fc_units,epochs,batch_size)
    
    return model