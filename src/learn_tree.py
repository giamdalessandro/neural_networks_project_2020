import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from maskLayer import *
from utils.visuallib import load_def

MASKED1 = "./models/masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5"

with tf.device("/CPU:0"):
    m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})
    print(m_trained.summary())

pred = m_trained.predict(load_def(
    folder="/media/luca/DATA2/uni/neural_networks_project_2020/dataset/train_val/bird/001.Black_footed_Albatross/", fileid="Black_Footed_Albatross_0001_796111.jpg"))
print("Classification score: {}".format(pred))


temp_model = Model(inputs=m_trained.input, outputs=m_trained.get_layer("flatten").output)
flatten_output = temp_model.predict(load_def(fileid="n02355227_obj/img/img/00010.jpg"))

def compute_g(model, inputs):
    '''
        Computes g = dy/dx, where x is the output of the top conv layer after the mask operation,
        and y is the output of the prediction before the softmax.
            - model: the pretrained modell on witch g will be computed;
            - imputs: x, the output of the top conv layer after the mask operation.
    '''
    fc_1 = model.get_layer("fc1")
    fc_2 = model.get_layer("fc2")
    fc_3 = model.get_layer("fc3")

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(fc_1.variables)      

        y = fc_3(fc_2(fc_1(inputs)))
        gradient = tape.gradient(y, fc_1.variables)

    return tf.reshape(tf.reduce_sum(gradient[0], axis=1), shape=(7,7,512))

print(compute_g(m_trained, flatten_output))

