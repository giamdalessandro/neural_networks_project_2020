import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# TODO
#   - Disentgled CNN
#       - aggiungere loss filtri
#   - Build decision trees


keras.applications.VGG16(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000, classifier_activation='softmax'
)

print("hello")