import os
import tensorflow as tf
from maskLayer import *

MASKED1 = "./models/masked1_binary_2_epochs_22_9_2020_17_19.h5"

imported = tf.keras.models.load_model(MASKED1, custom_objects={'MaskLayer':MaskLayer()})
print(imported.summary())