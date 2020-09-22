import os
import tensorflow as tf
import matplotlib.image as mpimg
from datetime import datetime as dt

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from matplotlib import pyplot as plt

from maskLayer import *

MASKED1 = "masked1_no_dropout_binary_50_epochs_21_9_2020_18_40.h5"
MASKED2 = "masked1_binary_50_epochs_21_9_2020_19_0.h5"


mdl1 = tf.keras.models.load_model(MASKED1, custom_objects={'MaskLayer':MaskLayer})
print(mdl1.summary)
