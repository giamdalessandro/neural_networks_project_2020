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

#   1. getting first 5 VGG16 pre-trained conv blocks without last max pool
VGG16 = keras.applications.VGG16(
    include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000, classifier_activation='softmax'
)
model = keras.Sequential(VGG16.layers[:-1])
print(model.summary())
print(">> Model loaded.")


#   2. modificare filtri nel top conv-layer --> aggiungere maschere
#   3. aggiungere un nuovo conv-layer con M=512 filtri --> ogni filtro è un tensore 3x3xM
#   4. aggiungere maschere per i filtri del nuovo conv-layer
#   5. usare gli stessi FC inizializzati random