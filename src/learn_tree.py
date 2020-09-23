import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from maskLayer import *
from utils.visuallib import load_def

MASKED1 = "./models/masked1_binary_25_epochs_22_9_2020_20_6.h5"

with tf.device("/CPU:0"):
    m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})
    print(m_trained.summary())

    #pred = m_trained.predict(load_def(fileid="n02355227_obj/img/img/00009.jpg"))
    #print("Classification score: {}".format(pred))


# to retrieve prediction mask_layer output 
intermediate_layer_model = Model(inputs=m_trained.input, outputs=m_trained.get_layer("mask_layer_1").output)
mask_output = intermediate_layer_model.predict(load_def())
print(mask_output)

