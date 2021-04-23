import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

# It can be used to reconstruct the model identically.
model = keras.models.load_model("conv")
model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.expand_dims(x_train, -1)

intermediate = keras.models.Model(model.input, model.get_layer('pool1').output)

result = intermediate(x_train[0:1]).numpy()
print(result.shape)

from PIL import Image
result = tf.reshape(result, (result.shape[3], result.shape[1], result.shape[2]))

result = result.numpy()

for i in range(result.shape[0]):
    Image.fromarray(result[i,:,:]).resize((result.shape[1], result.shape[2])).resize((300,300)).show()
    input()