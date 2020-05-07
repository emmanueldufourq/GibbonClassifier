# GibbonClassifier

In this study we describe the development of a classifier for identifying Hainan gibbon (Nomascus hainanus) calls in passive acoustic recordings collected as part of a long-term monitoring project whose conditions illustrate both the appeal and difficulty of automation. Hainan gibbons are one of the world's rarest mammal species, with fewer than 30 individuals believed to exist in the wild. 

Our goal was to develop an automated classifier for the monitoring project, together with software allowing the classifier to be run without deep learning or programming expertise. Below, we describe the usage of the software.

# Authors

Emmanuel Dufourq, Ian Durbach, James Hansford, Sam Turvey, Amanda Hoepfner

# Requirements

Install all requirements using `pip install -r requirements.txt`

Numpy

Scipy

Librosa

Pandas

Tensorflow

Sklearn

Pickle

Matplotlib

# Overview of approach

A brief overview of our approach is illustrated below.

<p align="center">
  <img src="https://github.com/emmanueldufourq/GibbonClassifier/blob/master/Overview.png?raw=true">
</p>

# Example Data Files

Download these two files to run through the example notebooks. Place the train file in Raw_Data/Train and the test file in Raw_Data/Test.

Train file: HGSM3D_0+1_20160429_051600.wav https://drive.google.com/open?id=1ELtriuMC0bXwSyOSjtlKrZzu_3B0BxH8

Test file: HGSM3B_0+1_20160308_055700.wav https://drive.google.com/open?id=14MtKQZsrecoQ_yIM_zFExgbJShOVOKsi

# Code pipeline

<p align="center">
  <img src="https://github.com/emmanueldufourq/GibbonClassifier/blob/master/Pipeline.jpg?raw=true">
</p>

The following two notebooks are available for execution. These can be run locally or on Google Colab.

1) `Train.ipynb` - extract audio segments from the original audio file, augment and preprocess. Train a CNN.

2) `Predict.ipynb` - predict using a trained model

# Executing code on Google Colab

Training on Google colab: https://colab.research.google.com/drive/12CBJcdsToGW7CKNT1GS0RSgI3TCJNFyt

Predicting on Google colab: https://colab.research.google.com/drive/16mfhu6STIkfv0UI3uuFWziQCoen014eT

# Additional files

1) `CNN_Network.py` - view/edit the CNN network

2) `Hyper_Parameters` - view/edit the CNN hyper-parameters 

3) `Train_Helper.py` - all functions which are used during training

4) `Predict_Helper.py` - all functions which are used for prediction

5) `Augmentation.py` - all functions which are used to augment the data

6) `Extract_Audio_Helper.py` - all functions which are used to extract and preprocess the audio files
