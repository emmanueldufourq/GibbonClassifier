import numpy as np
from CNN_Network import *

def check_clock(start, end, start_reading, end_reading, start_convert ,end_convert,
                start_model_loading, end_model_loading, 
                start_prediction,end_prediction):
                
    print ('Total execution time (seconds):', int(end-start))
    print ('\nBreak down:')
    print ('Time to read input file (seconds):', int(end_reading-start_reading))
    print ('Time to convert audio to spectrograms (seconds):', int(end_convert-start_convert))
    print ('Time to load CNN model (seconds):', int(end_model_loading-start_model_loading))
    print ('Time to perform predictions (seconds):', int(end_prediction-start_prediction))

def create_time_index(time_to_extract, file_duration_seconds):
    
    start = []
    end = []

    # Find out how many chunks of unit size (time_to_extract) can
    # be obtained 
    amount_of_chunks = int(file_duration_seconds - time_to_extract)
    
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

def post_process(predictions, threshold):

    values = predictions
    values = values[:,1]  > threshold
    values = values.astype(np.int)
    values = np.where(values == 1)[0]

    component_prediction = get_components(values)
    predict_components = check(component_prediction)
    predict_components = get_connected_components(predict_components, 0)
    
    return predict_components


def create_X_new(mono_data, time_to_extract, sampleRate, verbose):
    
    X_frequences = []
    segment_times = []

    sampleRate = sampleRate
    
    # Counter used to allocate unique name to each file
    counter = 0

    # Get the corresponding start time
    start_time_seconds = 0

    # Get the duration of the call
    duration = get_length_in_seconds(mono_data)

    # Compute end time
    end_time_seconds = start_time_seconds + duration

    if verbose:
        # Print spme info
        print ('-----------------------')
        print ('start (seconds)', start_time_seconds)
        print ('end (seconds)', end_time_seconds)
        print ('duration (seconds)', duration)
        print()

    # Find out how many chunks of unit size (time_to_extract) can
    # be obtained 
    amount_of_chunks = int(duration - time_to_extract)

    if verbose:
        print ('Chunks:',amount_of_chunks)
        
    # Create a list to store the y labels which will be generated below
    y_true = []
    
    # Iterate over each chunk to extract the frequencies
    for i in range (0, amount_of_chunks):
        
        # Locate the correct index values [start,end] based on the start time 
        # of the call and the chunk to use
        start_index = (i) 
        end_index = (time_to_extract + (i))
        
        call_present = False
                
        if verbose:
            print ('Index:', counter)
            print ('Chunk start time (sec):', start_index)
            print ('Chunk end time (sec):',end_index)
            
        # Extract the frequencies from the mono file
        extracted = mono_data[int(start_index *sampleRate) : int(end_index * sampleRate)]
        
        X_frequences.append(extracted)
        
        # Append start and end times
        start_end = np.array([start_index, end_index])
        segment_times.append(start_end)

        # increment counter so each file has a unique name
        counter = counter + 1
    
    X_frequences = np.array(X_frequences)

    if verbose:
        print ()

    return X_frequences


def get_length_in_seconds(librosa_audio):
    return len(librosa_audio)/4800

