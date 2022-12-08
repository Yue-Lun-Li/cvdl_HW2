from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import random , os, cv2
import matplotlib.pyplot as plt





IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

class Resnet50(object):
    def __init__(self):
        super(Resnet50, self).__init__()

    def model(self):
        input_shape = IMAGE_SIZE + (3,)
        inputs = keras.Input(shape=input_shape)
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        res_blocks = [3, 4, 6, 3]
        res_filters = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]

        first_conv = 1
        for index, block in enumerate(res_blocks):  # 0, 3
            for layer in range(block):  # 3
                input_tensor = x
                for idx, f in enumerate(res_filters[index]):
                    pad = 'valid'
                    ksize = (1, 1)
                    if idx > 0 and idx < 2:
                        ksize = (3, 3)
                        pad = 'same'

                    strides = (1, 1)
                    if first_conv == 1:
                        first_conv = 0

                    elif idx == 0 and layer == 0:
                        strides = (2, 2)

                    x = layers.Conv2D(f, ksize, strides=strides, kernel_initializer='he_normal', padding=pad)(x)
                    x = layers.BatchNormalization()(x)
                    if idx < 2:
                        x = layers.Activation("relu")(x)

                if layer == 0:
                    strides = (2, 2)
                    if index == 0:
                        strides = (1, 1)

                    shortcut = layers.Conv2D(res_filters[index][-1], (1, 1), strides=strides,
                                            kernel_initializer='he_normal')(input_tensor)
                    shortcut = layers.BatchNormalization()(shortcut)
                else:
                    shortcut = input_tensor

                x = layers.add([x, shortcut])
                x = layers.Activation('relu')(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        outputs = x
        resnet50 = keras.Model(inputs, outputs)
        return resnet50

def data_generator():
    train_datagen = ImageDataGenerator(rescale=1/255)
    val_datagen = ImageDataGenerator(rescale=1/255)
    test_generator  = val_datagen.flow_from_directory(
        'Dataset/test', 
        class_mode = 'binary', 
        target_size = IMAGE_SIZE, 
        batch_size = 1,
        shuffle = True
    )
    return test_generator

