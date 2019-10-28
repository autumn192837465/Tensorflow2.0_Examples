#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Diffient optimizer


# In[24]:


import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import optimizers


# In[25]:


# Get data
Data = pd.read_csv('iris.csv',header = None)


# Variables
[m,n] = Data.shape
fn = n - 1

trainNum = int(m * 0.7)

# Shuffle data
Data = Data.sample(frac=1).reset_index(drop=True)
# Seperate data into X y
DataX = Data.iloc[:,0:fn]
DataY = Data.iloc[:,fn]

# Seperate data into train test
trainX = DataX.iloc[0:trainNum,:].values
trainY = DataY.iloc[0:trainNum].values
testX = DataX.iloc[trainNum:,].values
testY = DataY.iloc[trainNum:,].values



# In[26]:


''' 
    commonly used activation : relu,selu,sigmoid,softmax,tanh    
    commonly used activation : sgd,adagrad,adam
'''
# Create SGD models 


model_SGD = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 10, 
                          activation = tf.nn.relu, 
                          input_shape=(4,), 
                          kernel_initializer = tf.initializers.RandomUniform()),
    tf.keras.layers.Dense(3,activation='softmax')
])
model_SGD.compile(optimizer = optimizers.SGD(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'],
                 )

sgd_acc = model_SGD.fit(trainX, trainY, epochs=1000,verbose=1)


# In[27]:


# Create Adagrad models
model_Adagrad = tf.keras.Sequential()

# Another way to add layer
model_Adagrad.add(tf.keras.layers.Dense(units = 10, 
                          activation = tf.nn.relu, 
                          input_shape=(4,), 
                          kernel_initializer = tf.initializers.RandomUniform()))
model_Adagrad.add(tf.keras.layers.Dense(units = 3, 
                          activation = tf.nn.softmax, 
                          kernel_initializer = tf.initializers.RandomUniform()))

model_Adagrad.compile(optimizer = optimizers.Adagrad(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

adagrad_acc = model_Adagrad.fit(trainX, trainY, epochs=1000,verbose=1)


# In[28]:


# Create Adam models
model_Adam = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 10, 
                          activation = tf.nn.relu, 
                          input_shape=(4,), 
                          kernel_initializer = tf.initializers.RandomUniform()
                         ),
    tf.keras.layers.Dense(3,activation='softmax')
])

model_Adam.compile(optimizer = "adam",
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

adam_acc = model_Adam.fit(trainX, trainY, epochs=1000,verbose=1)


# In[29]:


plt.figure();
plt.plot(sgd_acc.history["acc"],label ='SGD')
plt.plot(adagrad_acc.history["acc"],label ='Adagrad')
plt.plot(adam_acc.history["acc"],label ='Adam')

plt.legend(loc='center left',bbox_to_anchor=(1, 1))
test_loss, test_acc_SGD = model_SGD.evaluate(testX,  testY, verbose=0)
test_loss, test_acc_Adagrad = model_Adagrad.evaluate(testX,  testY, verbose=0)
test_loss, test_acc_Adam = model_Adam.evaluate(testX,  testY, verbose=0)

print('SGD test accuracy: {}'.format(test_acc_SGD))
print('Adagrad test accuracy: {}'.format(test_acc_Adagrad))
print('Adam test accuracy: {}'.format(test_acc_Adam))

