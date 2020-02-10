# Handwritten Equation Recognizer using CNN


This repository consists of Recognition of handwritten equations into their individual characters that is acheived using Convolutional Neural Network (CNN)

## Requirements for running

1. Python 3.6
2. Pytorch 1.4+

## Contents

1. Data Collection and Processing
2. Model Description


### 1. Data Collection and Processing

The initial data was collected from kaggle and due to data mismatch and redundant data some data were pruned and extracted data from MNIST database was added and collected into the training database. The point wise algorithm used in Data Processing is given below:
1. Import necessary Libraries and API (For this project, numpy, os, cv2, random, and matplotlib was imported)
2. Create a Label for each folder where the data is present. For our project there are 30 folders each with 24 unique characters used in our recognition system. Each folder consists of only the characters data which is denoted by the folder’s name. Hence for ‘-‘ image, the folder it is contained in is named ‘-‘. Store the labels in a variable with an identifier number (index).
3. Merge the image with the identifier using numpy and shuffle the data.
4. Now split the dataset into “HerX.npy” with all the image data and “HerY.npy” with all the labels for training.
5. Finally, make a separate numpy file “Labels_her.npy” with link to each identifier and image label. Example: For “1” our identifier is 5 so we link “1” with 5 so whenever 5 is predicted from the model, it is actually “1”. We do this for all 55 characters.

### 2. Model Description

