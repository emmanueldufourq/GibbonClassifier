import time
import numpy as np
import pandas as pd
import soundfile as sf
from os import listdir
import librosa
import collections
import time
import tarfile
import pandas as pd
import soundfile as sf
from os import listdir
import librosa
import collections
from CNN_Network import *
from Augmentation import convert_to_image

def execute_processing(testing_folder, testing_file, sample_rate,
                      location_model, weights_name, time_to_extract, prediction_folder):
    
    
    start = time.time()
    start_reading = time.time()
    
    print ('Reading audio file (this can take some time)...')
    test_file_audio, test_file_sample_rate = librosa.load(testing_folder + testing_file, 
                                                          sr=sample_rate) 
    
    print ()
    print ('Reading done.')
    end_reading = time.time()

    model_prediction = execute_batches(test_file_audio, time_to_extract, sample_rate, location_model, weights_name)
    
    start_times, end_times = create_time_index(time_to_extract, int(len(test_file_audio)/test_file_sample_rate))
    results = pd.DataFrame(np.column_stack((start_times, end_times, model_prediction[:,0],model_prediction[:,1])), 
                 columns=['Start(seconds)', 'End(seconds)', 'Pr(absence)', 'Pr(presence)'])
    
    np.savetxt(prediction_folder + testing_file + '_prediction.txt',model_prediction, fmt='%5f')
    results.to_csv(prediction_folder + testing_file + '_probabilities.txt', index=False)
    
    segments = post_process(model_prediction, 0.76)
    end_prediction = time.time()
    end = time.time()

    print ('---------------------------------------------------')
    print ('Predicted segment start and end times:')
    print (segments)
    
def create_time_index(time_to_extract, file_duration_seconds):
    
    start = []
    end = []

    # Find out how many chunks of unit size (time_to_extract) can
    # be obtained 
    amount_of_chunks = int(file_duration_seconds - time_to_extract+1)
    
    # Iterate over each chunk to extract the frequencies
    for i in range (0, amount_of_chunks):
        start.append(i)
        end.append(time_to_extract + (i))
    
    return np.array(start), np.array(end)

def get_components(values):
    shifted = np.roll(values,1)
    shifted[0] = 0
    difference = shifted - values
    shifted[difference < -200] = 0
    
    connected_component = []
    i = 0
    while i < len(shifted):
        #print ('i', i)
        if shifted[i] > 0:
            component = []
            j=0
            while j < len(shifted)-i:
                if shifted[i+j] > 0:
                    component.append(shifted[i+j])
                else:
                    break
                j = j + 1
            connected_component.append(component)
            i=i+j
            
        i = i+1
        
    return connected_component

def get_connected_components(components, verbose):
    gibbon_indices = []
    for component in components:
        if verbose:
            print ('Start ',component[0])
            print ('End ',component[-1]+10)
        gibbon_indices.append([component[0],component[-1]+10])
        
    return gibbon_indices

def check(preds):
    
    cleaned_components = []
    
    for component in preds:
        #print()
        #print (component)
        #print('len',len(component))
        
        rolled = component - np.roll(component,1)
        rolled[0] = 0
        #print ('average', np.average(rolled))
        
        if len(component) < 20:
            continue

        if np.average(rolled) < 10:
            #print('add')
            cleaned_components.append(component)
            
    return cleaned_components

def execute_batches(audio, time_to_extract, sample_rate, location_model, weights_name):
    
    batch_number = 8
    model_predictions = []
    start_index = 0
    end_index = 60*60
    
    for i in range(batch_number):
        
        print('Processing batch: {} out of {}'.format(i, batch_number))

        batch_prediction = process_batch(audio, start_index, end_index, 
                                         time_to_extract, sample_rate, location_model, weights_name)
        model_predictions.extend(batch_prediction)
        start_index = end_index - 9
        end_index = end_index + 60*60

    return np.array(model_predictions)

def process_batch(audio, start_index, end_index, time_to_extract, sample_rate, location_model, weights_name):
    

    # Extract segments from test file
    X = create_X_new(audio, 
                         time_to_extract, 
                         sample_rate,start_index, end_index, verbose = False)   
    
    # Convert data into spetrograms
    X = convert_to_image(X)
    
    # Build the model and load weights
    model = network()
    model.load_weights(location_model+weights_name)
    
    ## Predict
    model_prediction = model.predict(X, batch_size=128)
    
    return model_prediction

def post_process(predictions, threshold):

    values = predictions
    values = values[:,1]  > threshold
    values = values.astype(np.int)
    values = np.where(values == 1)[0]

    component_prediction = get_components(values)
    predict_components = check(component_prediction)
    predict_components = get_connected_components(predict_components, 0)
    
    return predict_components

def create_X_new(mono_data, time_to_extract, sampleRate,start_index, end_index, verbose):
    
    X_frequences = []

    sampleRate = sampleRate
    duration = end_index - start_index -9
    if verbose:
        # Print spme info
        print ('-----------------------')
        print ('start (seconds)', start_index)
        print ('end (seconds)', end_index)
        print ('duration (seconds)', (duration))
        print()
    counter = 0
    
    end_index = start_index + 10
    # Iterate over each chunk to extract the frequencies
    for i in range (0, duration):
    
        if verbose:
            print ('Index:', counter)
            print ('Chunk start time (sec):', start_index)
            print ('Chunk end time (sec):',end_index)
            
        # Extract the frequencies from the mono file
        extracted = mono_data[int(start_index *sampleRate) : int(end_index * sampleRate)]

        X_frequences.append(extracted)
        
        start_index = start_index + 1
        end_index = end_index + 1
        counter = counter + 1
        
    X_frequences = np.array(X_frequences)
    print (X_frequences.shape)
    if verbose:
        print ()

    return X_frequences


def get_length_in_seconds(librosa_audio, sample_rate):
    return int(len(librosa_audio)/sample_rate)

