import os
import tensorflow as tf
from maskLayer import *
from utils.dataset_utils import *
from utils.tree_utils import * 

MASKED1 = "./models/masked1_binary_25_epochs_22_9_2020_20_6.h5"

imported = tf.keras.models.load_model(MASKED1, custom_objects={'MaskLayer':MaskLayer()})
print(imported.summary())

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
