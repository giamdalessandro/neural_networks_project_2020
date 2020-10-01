import os
import json
import fnmatch
import numpy as np
import random as rd
import treelib as tl
import tensorflow as tf

from math import sqrt, log, exp
from scipy.optimize import minimize
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


L = 14*14
STOP = 10
FAKE = False
DTYPE = tf.float32
LAMBDA_0 = 0.000001
NUM_FILTERS = 512


######### OPERATIONS ###########

def optimize_g(g1, g2, fake=True):
    if fake:
        return tf.ones(shape=[512,1])

    from classes.interpretableNode import NUM_FILTERS
    g1 = tf.reshape(g1, shape=[512]).numpy()
    g2 = tf.reshape(g2, shape=[512]).numpy()

    b = (-1.0, 1.0)     # range nel quale può variare ogni elemento di g

    def objective(x):
        g_sum = np.add(g1, g2)
        return sum(-g_sum[0:]*x[0:])

    def constraint1(x):
        sum_eq = 1.0
        for i in range(NUM_FILTERS):
            sum_eq = sum_eq - x[i]**2.0
        return sum_eq

    x0 = np.zeros(NUM_FILTERS)

    bnds = np.full(shape=(NUM_FILTERS, 2), fill_value=b, dtype=float)
    cons = ([{'type': 'eq', 'fun': constraint1}])
    solution = minimize(objective, x0, bounds=bnds, constraints=cons)
    return tf.reshape(tf.convert_to_tensor(solution.x, dtype=DTYPE), shape=[512,1])


def vectorify_on_depth(x):
    """
    xx = tf.ones(shape=(2,2,5))
    x = tf.reduce_sum(xx, axis=[0,1])
    # sum over h and w --> outputs a vector of lenght d=5
    """
    return tf.reduce_sum(x, axis=[0, 1])


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

def compute_g(fc3_model, inputs):
    '''
    Computes g = dy/dx, where x is the output of the top conv layer after the mask operation,
    and y is the output of the prediction before the softmax.
        - model:  the pretrained modell on witch g will be computed;
        - imputs: x, the output of the top conv layer after the mask operation.
    '''
    fc_1 = fc3_model.get_layer("fc1")
    fc_2 = fc3_model.get_layer("fc2")
    fc_3 = fc3_model.get_layer("fc3")

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(fc_1.variables)

        y = fc_3(fc_2(fc_1(inputs)))
        gradient = tape.gradient(y, fc_1.variables)

    return tf.reshape(tf.reduce_sum(gradient[0], axis=1), shape=(7, 7, 512))

def sow(trained_model, pos_image_folder):
    """
    Sow the tree taking care of all its needs
    """
    from classes.interpretableTree import InterpretableTree

    start = dt.now()
    print("[TIME] -- sowing started ")
    tree = InterpretableTree()
    y_dict = tree.init_leaves(trained_model, pos_image_folder)
    x_dict = tree.vectorify(y_dict)
    tree.init_E()
    print("[TIME] -- sowing took  ", dt.now()-start)
    return tree, y_dict, x_dict


def grow(old_tree, y_dict, x_dict):
    from classes.interpretableTree import InterpretableTree
    start = dt.now()
    print("[TIME] -- growing started  ")
    
    t = 0             # different t indicates different trees in time

    while True:
        z = 1
        nid1 = None
        nid2 = None
        max_delta = None
        new_theta = 0
        new_node = None
        new_tree = InterpretableTree(s=old_tree.s,
                                     deep=True,
                                     theta=old_tree.theta,
                                     gamma=old_tree.gamma,
                                     tree=old_tree.subtree(old_tree.root))
        second_layer = old_tree.children(old_tree.root)

        num_couples = int(len(second_layer)*(len(second_layer)-1)/2)
        tested = 0
        
        for v1 in second_layer:
            start2 = dt.now()
            if z < len(second_layer):
                for v2 in second_layer[z:]:

                    tag = str(t)+"_"+str(tested)
                    node = new_tree.try_pair(v1, v2, tag=tag)
                    delta, theta = old_tree.compute_delta(node, v1, v2)
                    delta = delta.numpy()[0][0]
                    print("delta = ", delta)
                    if max_delta is None:
                        max_delta = delta
                    if max_delta is not None and delta > max_delta:  
                        nid1 = v1
                        nid2 = v2
                        new_node = node
                        new_theta = theta
                        max_delta = delta
                        
                    new_tree.ctrlz(v1, v2)
                    tested += 1
                    if tested % 10 == 0:
                        print("       >> tested couples :", tested, "on", num_couples)
            z += 1
            print("       >> tested couples :", tested, "on", num_couples, "in ", dt.now()-start2)


        if len(second_layer) == 1:
            print("len(second_layer) == 1")
            break
        print("       >> best delta     :", max_delta)
        if max_delta <= 0 or new_node is None:
            break
        
        t += 1
        new_tree.parentify(pid=new_node, nid1=nid1, nid2=nid2, theta=new_theta)
        new_tree.show()
        
    print("[TIME] -- growing took ", dt.now()-start)
    return new_tree


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
                    b=data["b"] if isinstance(data["b"],int) else str_to_tensor(data["b"]),
                    g=str_to_tensor(data["g"]),  
                    w=str_to_tensor(data["w"]),
                    x=str_to_tensor(data["x"]),
                    l=data["l"],
                    exph_val=data["exph"])

    if "children" not in current[curr_tag].keys():
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

    res_tree.E = dict_tree["tree_data"]["E"]
    res_tree.s = dict_tree["tree_data"]["s"]
    res_tree.theta = dict_tree["tree_data"]["theta"]
    res_tree.gamma = dict_tree["tree_data"]["gamma"]
    __parse_json_tree(res_tree, dict_tree, parent=None)

    res_tree.show()
    return res_tree
