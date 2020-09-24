import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from maskLayer import *
from utils.visuallib import load_def

MASKED1 = "./models/masked1_binary_25_epochs_22_9_2020_20_6.h5"

with tf.device("/CPU:0"):
    m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})
    #print(m_trained.summary())

    pred = m_trained.predict(load_def(fileid="n02355227_obj/img/img/00009.jpg"))
    print("Classification score: {}".format(pred))


# to retrieve prediction mask_layer output 
#intermediate_layer_model = Model(inputs=m_trained.input, outputs=m_trained.get_layer("max_pool").output)
#mask_output = intermediate_layer_model.predict(load_def())
#print(intermediate_layer_model.summary())

temp_model = Model(inputs=m_trained.input, outputs=m_trained.get_layer("flatten_1").output)
flatten_output = temp_model.predict(load_def(fileid="n02355227_obj/img/img/00010.jpg"))

def compute_g(model, inputs):
    '''
        Computes g = dy/dx, where x is the output of the top conv layer after the mask operation,
        and y is the output of the prediction before the softmax.
            - model: the pretrained modell on witch g will be computed;
            - imputs: x, the output of the top conv layer after the mask operation.
    '''
    fc_1 = model.get_layer("dense_3")
    fc_2 = model.get_layer("dense_4")
    #fc_3 = model.get_layer("dense_5")

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(fc_1.variables)      

        y = fc_2(fc_1(inputs))
        gradient = tape.gradient(y, fc_1.variables)

    return tf.reduce_sum(gradient[0], axis=None)

print(compute_g(m_trained, flatten_output))

'''
with tf.GradientTape() as tape:
   fc_output = network_up_to_fc(x)
   y = fully_connected_layer(fc)
   y_summed = tf.reduce_sum(y)
   gradient = tape.gradient(y, fc_output)
'''

''' HOW TO OBTAIN s
POSITIVE_IMAGE_SET = "./dataset/train_val/bird"
l = []  # len(POSITIVE_IMAGE_SET) x 512
for i in POSITIVE_IMAGE_SET:
    imported.predict(load_image())
    val = vectorify_on_depth(x)
    val = val / 14*14 (?????????)
    l.append(val)
val = avg(list, axis=0)
val = val / len(list)
'''