# Urban-Sound-Classification

## Introduction

Our main goal is to be able to split the environmental and urban sounds into 6 categories, i.e. dog barking, jackhammer, engine idling, children playing, street music, air conditioner. We employ three different machine learning methods in order to determine which category they belong to. **_K-Nearest Neighbors with Euclidian and Manhattan distance_**, **_Logistic Regression_** and **_Random Forest_** are the methods used in this problem.

## Selected Dataset

Our dataset is [“URBANSOUND8K”](https://urbansounddataset.weebly.com/urbansound8k.html). This dataset contains 8732 labeled sound excerpts in WAV format, whose duration are less than 4 seconds. We believe that this relatively short duration is enough for the scope this project and the amount of data that we have, 8732, which we consider as a good starting point. The sound excerpts belong to the following 10 classes: Air conditioner, Car horn, Children playing, Dog bark, Drilling, Engine idling, Gun shot, Jackhammer, Siren, Street music.

## Feature Extraction and Data Preprocessing

This dataset was used in a classification experiment as explained in [the paper](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf). We have reviewed this paper and the _**Librosa**_ library and decided to use the following features to train our dataset. One may extract other features with the Librosa library, however, we believe that 4 features suffice.

- Mel-Frequency Cepstral Coefficients
- Zero Crossing Rate
- Spectral Centroid
- Spectral Roll-Off

To reduce the complexity and computation cost, we reduce the number of classes to 6. We choose the following classes to continue.
- Dog bark
- Jackhammer
- Engine idling
- Children playing
- Street music
- Air conditioner

For smaller computation cost and more robust system, we need to do some data preprocessing before we train our models with dataset. The given dataset contains redundant instances and classes. With 6 classes that we choose above, the dimension of the features must be reduced. For this purpose, we use _**Principal Component Analysis (PCA)**_. PCA projects the feature vectors onto the lower dimensional vectors. It aims to maximize the variance on projected vector. 

The data preprocessing steps that we employ are as follows. We first take 1000 samples from each class. Note that there are 6 classes and we have equal number of samples per class. Secondly, we extract the features using the Librosa library. Later, we separate this dataset into the training set and the test set. We preserve 720 samples per class for training, 90 samples per class for test and 90 samples per class for validation. The dimensionality of the feature vector without PCA is 27. Then, we apply PCA and reduce the dimensionality to 21. We use these final sets for our methods. As the dataset have already prepared in 10 folds, we do _**10-fold cross validation**_.
