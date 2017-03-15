# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:57:49 2017

@author: Jing Wang
"""

from tensorflow.examples.tutorials.mnist import input_data
from numpy import concatenate, mean, asarray
from svmutil import *
import time

# load the MNIST data by TensorFlow
mnist=input_data.read_data_sets("MNIST_data/", one_hot=False)

image_train=mnist.train.images
image_validation=mnist.validation.images
image_test=mnist.test.images

label_train=mnist.train.labels
label_validation=mnist.validation.labels
label_test=mnist.test.labels

# merge the training and validation datasets
image_train=concatenate((image_train, image_validation), axis=0)
label_train=concatenate((label_train, label_validation), axis=0)

# array to list
image_train=image_train.tolist()
image_test=image_test.tolist()
label_train=label_train.tolist()
label_test=label_test.tolist()

# record time
time_start=time.time() 

# linear SVM by Libsvm in Python
model=svm_train(label_train,image_train,'-t 0')
label_predict, accuracy, decision_values=svm_predict(label_test,image_test,model)

# int to float
label_predict=[int(tmp) for tmp in label_predict] 

# list to array
label_predict=asarray(label_predict)
label_test=asarray(label_test)

# accuracy
accuracy=mean((label_predict==label_test)*1)
print('Accuracy: %0.4f.' % accuracy)

# time used
time_end=time.time()
print('Time to classify: %0.2f minuites.' % ((time_end-time_start)/60))

# # Output:
# Accuracy = 93.99% (9399/10000) (classification)
# Accuracy: 0.9399. 
# Time to classify: 13.21 minuites.
