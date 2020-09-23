import os
import tensorflow as tf
from maskLayer import *

MASKED1 = "./models/masked1_binary_25_epochs_22_9_2020_20_6.h5"


with tf.device("/CPU:0"):
    imported = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})
    print(imported.summary())