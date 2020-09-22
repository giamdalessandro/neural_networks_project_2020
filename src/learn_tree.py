import os
import tensorflow as tf
import matplotlib.image as mpimg
from datetime import datetime as dt

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from matplotlib import pyplot as plt

from maskLayer import *

MASKED1 = "./models/masked1_binary_2_epochs_22_9_2020_12_18"
MASKED2 = "masked1_binary_50_epochs_21_9_2020_19_0.h5"


mdl1 = tf.keras.models.load_model(MASKED1, custom_objects={'MaskLayer':MaskLayer()})
print(mdl1.summary)

# we need to use these lines to update the custom object scope
''' 
# At loading time, register the custom objects with a `custom_object_scope`:
custom_objects = {"CustomLayer": CustomLayers}
with keras.utils.custom_object_scope(custom_objects):
    new_model = keras.Model.from_config(config)
'''
