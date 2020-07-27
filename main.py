import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# GPU check
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

model = keras.applications.VGG16(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000, classifier_activation='softmax'
)
print(model.summary())

layer = model.get_layer(name='block1_conv1')
print(layer.weights)