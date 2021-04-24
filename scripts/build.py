import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

input_shape = (28, 28, 1)
num_classes = 10

inputs = keras.Input(input_shape)
lastLayer = inputs
for i in range(4):
    li = []
    for size in [3, 5]:
        layer = layers.Conv2D(
            1, size, padding='same', activation='relu'
        )(lastLayer)
        for feature in (layer[:,:,:, i] for i in range(layer.shape[3])):
            li.append(feature)

    li.append(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(lastLayer)[:,:,:,0])
    conv = tf.stack(li, 3)
    lastLayer = conv

avgpool = layers.AveragePooling2D(
    pool_size=(7, 7)
)(lastLayer)
flatten = layers.Flatten()(avgpool)
outputs = layers.Dense(num_classes, activation=tf.nn.softmax)(flatten)

model = keras.models.Model(inputs, outputs)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

model.save('conv')