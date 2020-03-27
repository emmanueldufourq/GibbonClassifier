# GibbonClassifier

In this study we describe the development of a classifier for identifying Hainan gibbon (Nomascus hainanus) calls in passive acoustic recordings collected as part of a long-term monitoring project whose conditions illustrate both the appeal and difficulty of automation. Hainan gibbons are one of the world's rarest mammal species, with fewer than 30 individuals believed to exist in the wild. Our goal was to develop an automated classifier for the monitoring project, together with software allowing the classifier to be run without deep learning or programming expertise. Below, we describe the usage of the software.

# Authors

Emmanuel Dufourq, Ian Durbach, James Hansford, Sam Turvey, Amanda Hoepfner

# Requirements

Numpy
Scipy
Librosa
Pandas
Keras


# Example Data Files

Download these two files to run through the example notebooks. Place the train file in Raw_Data/Train and the test file in Raw_Data/Test.

Train file: HGSM3D_0+1_20160429_051600.wav https://drive.google.com/open?id=1ELtriuMC0bXwSyOSjtlKrZzu_3B0BxH8

Test file: HGSM3B_0+1_20160308_055700.wav https://drive.google.com/open?id=14MtKQZsrecoQ_yIM_zFExgbJShOVOKsi

# Code pipeline

![Software Pipeline](https://github.com/emmanueldufourq/GibbonClassifier/Pipeline.jpg)
# Usage

Once the training and test files have been downloaded the pipeline is executed as follows

1) `Extract_Audio notebook.ipynb` - extract audio segments from the original audio file

2) `Augmentation_Execution.ipynb` - augment the number of calls

3) `Train.ipynb` - train a CNN

4) `Prediction.ipynb` - predict using a trained model

5) `Post_Processing.ipynb` - apply post processing heuristic

# Additional files

1) `CNN_Network.py` - view/edit the CNN network

2) `Hyper_Parameters` - view/edit the CNN hyper-parameters 

3) `No_Post_Processing` - get network output in seconds instead of binary/softmax values
