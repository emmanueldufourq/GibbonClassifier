import numpy as np

def get_components(values):
    shifted = np.roll(values,1)
    shifted[0] = 0
    difference = shifted - values
    shifted[difference < -200] = 0
    
    connected_component = []
    i = 0
    while i < len(shifted):
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

        rolled = component - np.roll(component,1)
        rolled[0] = 0
        
        if len(component) < 20:
            continue

        if np.average(rolled) < 10:
            cleaned_components.append(component)
            
    return cleaned_components


def heuristic(files, folder_location):

    for file in files:

        values = np.loadtxt(folder_location+file+'.wav_binary_prediction.txt').astype(np.int64)
        values = values[:,1]  > 0.76
        values = values.astype(np.int)
        values = np.where(values == 1)[0]

        component_prediction = get_components(values)
        component_correct = get_components(correct)
    
        predict_components = check(component_prediction)
        predict_components = get_connected_components(predict_components, 0)
        correct_components = get_connected_components(component_correct, 0)

        np.savetxt('../Predictions/{}_post_processing_heuristic.txt'.format(file), np.asarray(predict_components), fmt='%i') 
        print ('Saved post-processing in location: /Predictions/{}_post_processing_heuristic.txt'.format(file))
        print ()
