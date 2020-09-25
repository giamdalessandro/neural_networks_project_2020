import os
import fnmatch
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from classes.maskLayer import MaskLayer
from classes.decisionTree import *
from utils.dataset_utils import load_test_image

MODELS  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")


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

    return tf.reshape(tf.reduce_sum(gradient[0], axis=1), shape=(7, 7, 512))


def initialize_leaves(trained_model, tree, pos_image_folder=POSITIVE_IMAGE_SET):
    """
    Initializes a root's child for every image in the positive image folder 
    """
    root = tree.create_node(tag="root", identifier='root')


    flat_model = Model(inputs=trained_model.input,outputs=trained_model.get_layer("flatten").output)
    fc3_model = Model(inputs=trained_model.input,outputs=trained_model.get_layer("fc3").output)

    s_list = []
    gamma = 0
    pos_image_folder = os.path.join(pos_image_folder, 'test')   # we test only on a subset of images (10 images)
    
    for img in os.listdir(pos_image_folder):
        if img.endswith('.jpg'):
            test_image  = load_test_image(folder=pos_image_folder, fileid=img)
            flat_output = flat_model.predict(test_image)
            fc3_output  = fc3_model.predict(test_image)

            g = compute_g(fc3_model, flat_output)

            x = tf.reshape(flat_output, shape=(7,7,512))
            b = tf.subtract(fc3_output, tf.reduce_sum(tf.math.multiply(g, x), axis=None))   # inner product between g, x
            s = tf.math.reduce_mean(x, axis=[0,1])
            s_list.append(s)
            gamma += fc3_output[0]
            tree.create_node(tag=img[:-4], identifier=img[:-4], parent='root', g=g, alpha=tf.ones(shape=512), b=b)
    
    cardinality = len(fnmatch.filter(os.listdir(pos_image_folder), '*.jpg'))    # number of images in the positive set
    root.b = tf.reduce_sum(s_list, axis=0)/cardinality                          # sum on s to obtain s(1,1,512)
    root.l = cardinality/gamma


#with tf.device("/CPU:0"):
m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})
print(m_trained.summary())
tree = DecisionTree()
initialize_leaves(m_trained, tree)
tree.show()
for i in tree.all_nodes():
    i.print_info()
