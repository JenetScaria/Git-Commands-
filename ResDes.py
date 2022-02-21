#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

import tensorflow as tf
from tensorflow import keras


# In[2]:


Dense = tf.keras.layers.Dense
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
MaxPool2D = tf.keras.layers.MaxPool2D
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
Input = tf.keras.layers.Input
Activation = tf.keras.layers.Activation
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
BatchNormalization = tf.keras.layers.BatchNormalization
Concatenate= tf.keras.layers.Concatenate

arch = tf.keras.models.Model

tf.keras.backend.set_image_data_format('channels_last')


# In[3]:


DenseNet121 = tf.keras.applications.densenet.DenseNet121
Densenet169 = tf.keras.applications.densenet.DenseNet169
Densenet201 = tf.keras.applications.densenet.DenseNet201

ResNet50 = tf.keras.applications.resnet.ResNet50
ResNet101 = tf.keras.applications.resnet.ResNet101

ResNet50V2 = tf.keras.applications.resnet_v2.ResNet50V2
ResNet101V2 = tf.keras.applications.resnet_v2.ResNet101V2


# In[4]:


def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


# In[5]:


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 #if backend.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


# In[6]:


def model_one_class(
        input_shape = (224,224,3),
        class_6=6,
        class_20=20,
        class_82=75):
    # for results of sota papers
    inp = Input(input_shape)
    base_model= DenseNet121(include_top=False, weights=None, input_tensor = inp, backend = tf.keras.backend , layers = tf.keras.layers , models = tf.keras.models , utils = tf.keras.utils)
    
    x =  base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(class_82, activation='softmax')(x)

    model = arch(inputs=inp, outputs= [x])

    for layer in base_model.layers:
        layer.trainable = True
    
    return model


# In[ ]:




