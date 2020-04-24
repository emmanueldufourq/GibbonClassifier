import pandas as pd
import librosa
from scipy import signal
import numpy as np
import scipy
import matplotlib.pyplot as plt

import struct
from scipy.signal import argrelextrema
from scipy.fftpack import rfft, irfft, fftfreq
import time

import librosa.display
import scipy.fftpack

def read_and_process_gibbon_timestamps(directory, file_name, sample_rate, sep):
    ''' Read in a single file containing the timestamps of the gibbon calls
    and returns a Pandas dataframe with the start and end times converted
    using the sampling rate of the original file. The column `Type` denotes
    the number of gibbon pulses.
    
    '''
    gibbon_timestamps = pd.read_csv(directory+file_name, sep=sep)
    gibbon_timestamps.Notes.fillna(gibbon_timestamps.Type.astype(str), inplace=True)
    gibbon_timestamps['Notes'] = gibbon_timestamps.Notes.str.extract('(\d+)')
    gibbon_timestamps['Start'] = gibbon_timestamps['Start'] * sample_rate
    gibbon_timestamps['End'] = gibbon_timestamps['End'] * sample_rate
    gibbon_timestamps.drop(['Type'], axis=1, inplace=True)
    gibbon_timestamps.columns = ['Start', 'End',' Duration','Type']
    return gibbon_timestamps

def read_and_process_nongibbon_timestamps(directory, file_name, sample_rate, sep):
    ''' Read in a single file containing the timestamps of the background 
    noise and returns a Pandas dataframe with the start and end times converted
    using the sampling rate of the original file. The column `Type` denotes
    the type of background noise.
    
    '''
    non_gibbon_timestamps = pd.read_csv(directory+file_name, sep=sep)
    non_gibbon_timestamps['Start'] = non_gibbon_timestamps['Start'] * sample_rate
    non_gibbon_timestamps['End'] = non_gibbon_timestamps['End'] * sample_rate    
    return non_gibbon_timestamps

def extract_all_gibbon_calls(librosa_audio, gibbon_timestamp_df = None, alpha=10, 
                             jump_seconds=1 ,sample_rate = 0, verbose=0):

    '''
    Extract all of the gibbon calls based on a sliding window approach.
    The size of the window is defied as alpha.
    
    jump_seconds dictactes how the sliding window is moved when extracting data
    The units of the jump_seconds is in seconds
    A large jump_seconds means that fewer data points will be extracted.
    A small jump_seconds means that more data points will be extracted
    
    The function returns a Numpy array which contains all of the extracted
    gibbon calls.
    '''

    
    alpha_converted = alpha * sample_rate
    gibbon_extracted = []

    for index, row in gibbon_timestamp_df.iterrows(): 
        jump = 0

        # Keep trying to extract the gibbon call by shifting a small amount
        # to the left starting from the initial timestamp for the state of
        # the call. A jump to the left is taken and this jump can be controlled
        # using the jump_rate. On each iteration a certain amount of data is 
        # extracted (based on alpha * sample_rate). When this value is less
        # than the end of the call then stop extracting data.
        while True:
            start_position = row['Start'] - sample_rate - (jump * jump_seconds * sample_rate)
            end_position = start_position + alpha_converted
            jump = jump + 1

            if verbose:
                print ('start_position',start_position)
                print ('end_position',end_position)
                print ()
            if end_position <= row['End']:
                if verbose:
                    print('Breaking.')
                break
                
             # Append the audio data
            gibbon_extracted.append(librosa_audio[int(start_position):int(end_position)])

    return np.asarray(gibbon_extracted)

def extract_all_nongibbon_calls(librosa_audio, non_gibbon_timestamps = None,alpha=10, 
                                jump_seconds=1 ,sample_rate = 0, verbose=0):

    '''
    Extract all of the background noise based on a sliding window approach.
    The size of the window is defied as alpha.
    
    jump_seconds dictactes how the sliding window is moved when extracting data
    The units of the jump_seconds is in seconds
    A large jump_seconds means that fewer data points will be extracted.
    A small jump_seconds means that more data points will be extracted
    
    The function returns a Numpy array which contains all of the extracted
    background noise.
    '''

    
    alpha_converted = alpha * sample_rate
    noise_extracted = []

    for index, row in non_gibbon_timestamps.iterrows(): 
        jump = 0

        # Keep trying to extract the gibbon call by shifting a small amount
        # to the left starting from the initial timestamp for the state of
        # the call. A jump to the left is taken and this jump can be controlled
        # using the jump_rate. On each iteration a certain amount of data is 
        # extracted (based on alpha * sample_rate). When this value is less
        # than the end of the call then stop extracting data.
        while True:
            start_position = row['Start'] + (jump * jump_seconds * sample_rate)
            end_position = start_position + alpha_converted
            jump = jump + 1

            if verbose:
                print ('start_position',start_position)
                print ('end_position',end_position)
                print ()
            if end_position >= row['End']:
                if verbose:
                    print('Breaking.')
                break
                
             # Append the audio data
            noise_extracted.append(librosa_audio[int(start_position):int(end_position)])

    return np.asarray(noise_extracted)

