#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[10]:


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './cats_and_dogs_filtered/train',
    target_size = (384,384),
    shuffle=True,
    batch_size = 1,
    class_mode = 'binary'
)

test_generator = test_datagen.flow_from_directory(
    './cats_and_dogs_filtered/validation',
    target_size = (256,256),
    shuffle=True,
    batch_size = 50,
    class_mode = 'binary'
)


# In[8]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,3, padding = 'same',input_shape = (384,384, 3), activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(0.005)),
    tf.keras.layers.MaxPooling2D(3),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64,2, padding = 'same',activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64,2, padding = 'same',activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128,2, padding = 'same',activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128,2, padding = 'same',activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPooling2D(),    
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(1,activation = 'sigmoid')
])
model.compile(loss = 'binary_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])


# In[9]:


model.summary()


# In[12]:


model.fit_generator(train_generator,
                    
                    epochs=100,
                    validation_data = test_generator,
                    )


# In[ ]:




