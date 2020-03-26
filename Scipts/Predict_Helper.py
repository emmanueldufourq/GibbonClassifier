import numpy as np
from CNN_Network import *


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

