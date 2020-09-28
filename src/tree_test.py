import os
import fnmatch
import numpy as np
import tensorflow as tf

from math import log
from datetime import datetime as dt
from tensorflow.keras.models import Model

from classes.maskLayer import MaskLayer
from classes.interpretableTree import *
from utils.dataset_utils import load_test_image

MODELS  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")
TEST = False

'''
def compute_g(model, inputs):
    \'''
        Computes g = dy/dx, where x is the output of the top conv layer after the mask operation,
        and y is the output of the prediction before the softmax.
            - model: the pretrained modell on witch g will be computed;
            - imputs: x, the output of the top conv layer after the mask operation.
    \'''
    fc_1 = model.get_layer("fc1")
    fc_2 = model.get_layer("fc2")
    fc_3 = model.get_layer("fc3")

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(fc_1.variables)

        y = fc_3(fc_2(fc_1(inputs)))
        gradient = tape.gradient(y, fc_1.variables)

    return tf.reshape(tf.reduce_sum(gradient[0], axis=1), shape=(7, 7, 512))
'''

def initialize_leaves(trained_model, tree, pos_image_folder):
    """
    Initializes a root's child for every image in the positive image folder and returns the list of all predictions done
    """
    start = dt.now()
    root = tree.create_node(tag="root", identifier='root')

    flat_model = Model(inputs=trained_model.input,outputs=trained_model.get_layer("flatten").output)
    fc3_model = Model(inputs=trained_model.input,outputs=trained_model.get_layer("fc3").output)
    tree.fc3_model = fc3_model
    tree.flat_model = flat_model

    y_dict = {}
    s_list = []
    '''
    if TEST:
        pos_image_folder = os.path.join(pos_image_folder, 'test/bird')   # we test only on a subset of images (10 images)
    else:
        pos_image_folder = os.path.join(pos_image_folder, 'bird')   # 12k imgs
    '''
    # find . -type f -print0 | xargs -0 mv -t .
    # command to copy all filesf from subdirectories of the current directory in the current directory

    i = 0
    for img in os.listdir(pos_image_folder):
        if img.endswith('.jpg'):
            test_image  = load_test_image(folder=pos_image_folder, fileid=img)
            flat_output = flat_model.predict(test_image)
            fc3_output  = fc3_model.predict(test_image)[0][0]      # we take only the positive prediction score

            y_dict.update({img[:-4]:fc3_output})
            
            g = tree.compute_g(flat_output)
            x = tf.reshape(flat_output, shape=(7,7,512))
            b = tf.subtract(fc3_output, tf.reduce_sum(tf.math.multiply(g, x), axis=None))   # inner product between g, x
            
            s = tf.math.reduce_mean(x, axis=[0,1])
            s_list.append(s)
            tree.create_node(tag=img[:-4], identifier=img[:-4], parent='root', g=g, alpha=tf.ones(shape=512), b=b, x=x)
            
            i += 1
            if i % 10 == 0:
                print(">> created", i, "nodes")
            if i==STOP:
                break
            
            # TEST IF g and b ARE ACCURATE ENOUGH - IS WORKING! #
            # print("\nORIGINAL y -- CALULATED y")
            # print(fc3_output, " = ", tf.add(tf.reduce_sum(tf.math.multiply(g, x), axis=None), b).numpy())


    cardinality = len(fnmatch.filter(os.listdir(pos_image_folder), '*.jpg'))    # number of images in the positive set
    tree.s = tf.reduce_sum(s_list, axis=0)/cardinality                          # sum on s to obtain s(1,1,512)
    
    print("[TIME] -- initialize leaves took ", dt.now()-start)
    return y_dict


def grow(tree_0):
    """
    Grows the tree merging nodes until the condition is met
    """
    start = dt.now()
    curr_tree = tree_0
    e = 1
    p = 0                                               # index of the tree (P_i in the paper)
    while e > 0:
        curr_tree = curr_tree.choose_pair(tree_0, p)
        e = curr_tree.compute_delta()
        p += 1
        print("       >> generated tree:    ", p)
        #print("       >> delta E = ", e-e_0)

    print("[TIME] -- growing took           ", dt.now()-start)
    return curr_tree


#####################################################################################


with tf.device("/CPU:0"):
    m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})

STOP = 10
NEG_IMAGE_SET_TEST = "./dataset/train_val/test/bird/"
POS_IMAGE_SET_TEST = "./dataset/train_val/test/not_bird/"
#STOP = 12454

tree = InterpretableTree()
y_dict = initialize_leaves(m_trained, tree, POS_IMAGE_SET_TEST)   # initializes a leaf forall image in the positive set with the right parameters
tree.vectorify(y_dict)            # updates value (must be called after initialize_leaves())
if STOP < 20:
    tree.show()
new_tree = grow(tree)
new_tree.info()
if STOP < 20:
    new_tree.show()
saved = new_tree.save2json(save_name="test_tree")
loaded = InterpretableTree.from_json("./forest/test_tree.json")


'''
TODO
    - scrivere "find_gab"       - ro
    - scrivere "e_func"         - balthier
    - calcolare matrice A       - ???
'''

