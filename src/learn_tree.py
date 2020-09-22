import os
import tensorflow as tf
from maskLayer import *

MASKED1 = "./models/test_1"

#tf.keras.utils.register_keras_serializable(package="MaskLayer", name=None)

imported = tf.keras.models.load_model(MASKED1) #, custom_objects={'MaskLayer':MaskLayer()})
print(imported.summary())

# we need to use these lines to update the custom object scope
''' 
# At loading time, register the custom objects with a `custom_object_scope`:
custom_objects = {"CustomLayer": CustomLayers}
with keras.utils.custom_object_scope(custom_objects):
    new_model = keras.Model.from_config(config)
'''
