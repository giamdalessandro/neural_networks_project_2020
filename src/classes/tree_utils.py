import os
import json
import fnmatch
import numpy as np
import random as rd
import treelib as tl
import tensorflow as tf

from math import sqrt, log, exp
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


L = 14*14
STOP = 10
DTYPE = tf.int32
NUM_FILTERS = 512
LAMBDA_0 = 0.000001
NEG_IMAGE_SET_TEST = "./dataset/train_val/test/bird/"
POS_IMAGE_SET_TEST = "./dataset/train_val/test/not_bird/"



######### OPERATIONS ###########


def vectorify_on_depth(x):
    """
    xx = tf.ones(shape=(2,2,5))
    x = tf.reduce_sum(xx, axis=[0,1])
    # sum over h and w --> outputs a vector of lenght d=5
    """
    return tf.reduce_sum(x, axis=[0, 1])


def uguale(x1, x2):
    if tf.reduce_sum(tf.subtract(x1, x2)) == 0:
        return True
    else:
        return False


def load_test_image(folder, fileid):
    """
    loads and preprocesses default img specified in 'visualize.py' in variable 'path'
    """
    path = os.path.join(folder, fileid)
    img = load_img(path, target_size=(224, 224))      # test image
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def sow(trained_model, pos_image_folder):
    """
    Sow the tree taking care of all its needs
    """
    tree = InterpretableTree()
    y_dict = tree.init_leaves(trained_model, pos_image_folder)
    tree.vectorify(y_dict)


def grow(tree_0):
    """
    Grows the tree merging nodes until the condition is met
    """
    start = dt.now()
    curr_tree = tree_0
    e = 1
    # index of the tree (P_i in the paper)
    p = 0
    while e > 0:
        curr_tree = curr_tree.choose_pair(tree_0, p)
        e = curr_tree.compute_delta()
        p += 1
        print("       >> generated tree:    ", p)
        #print("       >> delta E = ", e-e_0)

    print("[TIME] -- growing took           ", dt.now()-start)
    return curr_tree



######### SAVE & LOAD ###########


def str_to_tensor(str_val, dtype="float32"):
    """
    Converts string np.array to tf.Tensor
    """
    np_val = np.array(str_val.strip('[]\n').split(), dtype=dtype)
    return tf.convert_to_tensor(np_val)


def __parse_json_tree(tree, current, parent=None):
    """
    Parse a tree from a JSON object returned by from_json()
        - tree      : tree instance where the parsed json_tree will be saved 
        - current   : node to parse, initially the JSON tree returned by from_json() 
        - parent    : parent node of current, initially None
    """
    par_tag = list(parent.keys())[0] if parent is not None else parent
    curr_tag = current if isinstance(current,str) else list(current.keys())[0]
    # print("<On node ->", curr_tag)

    data = current[curr_tag]["data"]
    tree.create_node(tag=curr_tag, identifier=curr_tag, parent=par_tag, 
                    alpha=str_to_tensor(data["alpha"]),
                    g=str_to_tensor(data["g"]), 
                    b=data["b"] if isinstance(data["b"],int) else str_to_tensor(data["b"]), 
                    l=data["l"])
    if "children" not in current[curr_tag].keys(): #isinstance(current,str):
        # print(" | -- on leaf ", curr_tag)
        return 

    else:
        for child in current[curr_tag]["children"]:
            # print("-- on child ", child)
            __parse_json_tree(tree, current=child, parent=current)
            
    return


def from_json(res_tree, save_path):
    """
    Loads a tree from a JSON file
        - save_path : path to the JSON tree file to load
    """
    with open(save_path, "r") as f:
        dict_tree = json.load(f)
        #print(dict_tree)

    __parse_json_tree(res_tree, dict_tree, parent=None)

    res_tree.show()
    return res_tree
