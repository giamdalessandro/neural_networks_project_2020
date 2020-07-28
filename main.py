import tensorflow as tf
from tensorflow import keras
#import numpy as np
#import matplotlib.pyplot as plt

# TODO
#   - Disentgled CNN
#       - aggiungere loss filtri
#   - Build decision trees


# GPU check
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# getting first four VGG16 pre-trained conv blocks
VGG16 = keras.applications.VGG16(
    include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000, classifier_activation='softmax'
)
#print(VGG16.summary())

print("Model loaded.")

model = keras.Sequential(VGG16.layers[:-1])
print(model.summary())
