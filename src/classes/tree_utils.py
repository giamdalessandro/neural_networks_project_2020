import os
import json
import fnmatch
import numpy as np
import random as rd
import treelib as tl
import tensorflow as tf

from beepy import beep
from math import sqrt, log, exp
from scipy.optimize import minimize
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


L = 14*14
STOP = 100
FAKE = False
DTYPE = tf.float32
LAMBDA_0 = 0.000001
NUM_FILTERS = 512


######### OPERATIONS ###########

def optimize_g(g1, g2):
    """
    Computes the new g starting from g1 and g2 of nodes v1 and v2
        - g1:   tensorflow array
        - g2:   tensorflow array
    """

    from classes.interpretableNode import NUM_FILTERS
    g1 = tf.reshape(g1, shape=[NUM_FILTERS])
    g2 = tf.reshape(g2, shape=[NUM_FILTERS])

    b = (-1.0, 1.0)     # range nel quale può variare ogni elemento di g

    def objective(x):
        g_sum = tf.math.add(g1, g2)
        return tf.reduce_sum(tf.math.multiply(-1*g_sum, x)).numpy()

    def constraint1(x):
        sum_eq = 1.0
        return (sum_eq - tf.reduce_sum(tf.math.multiply(x,x)).numpy())

    x0 = np.zeros(NUM_FILTERS)

    bnds = np.full(shape=(NUM_FILTERS, 2), fill_value=b, dtype=float)
    cons = ([{'type': 'eq', 'fun': constraint1}])
    solution = minimize(objective, x0, bounds=bnds, constraints=cons)
    return tf.reshape(tf.convert_to_tensor(solution.x, dtype=DTYPE), shape=[512,1])


def optimize_alpha(Xs, Ys, l=LAMBDA_0):
    '''
    Execute the LASSO problem to find the best value for alpha
        - Xs = list of g°x
        - Ys = list of y AFTER the softmax
    '''
    from sklearn.linear_model import Lasso

    lasso = Lasso(l, max_iter=5000)
    lasso.fit(Xs, Ys)
    coeff_used = np.sum(lasso.coef_ != 0)
    alpha = np.zeros(shape=[512])
    for þ in lasso.sparse_coef_.nonzero()[1]:
        alpha[þ] = 1

    return tf.reshape(tf.convert_to_tensor(alpha, dtype=DTYPE), shape=[512,1])


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

def IDentify(nid1, nid2):
    """
    Creates a unique ID such that follows this rule: when merging two nodes, the new node will have the combined ID of the two in lexographical order
    """
    return nid1+'#_#'+nid2 if nid1 < nid2 else nid2+'#_#'+nid1

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
    """
    Grow the tree taking care of all its needs
    """
    from classes.interpretableTree import InterpretableTree
    start = dt.now()
    print("[TIME] -- growing started  ")
    
    t = 0             # different t indicates different trees in time
    nodes_dict = {}

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
        start2 = dt.now()
        for v1 in second_layer:
            if z < len(second_layer):
                for v2 in second_layer[z:]:

                    tag = str(t)+"_"+str(tested)
                    nid = IDentify(v1.identifier, v2.identifier)
                    
                    if nid in nodes_dict:
                        node = retrieve_node(nid, node, nodes_dict, v1, v2, new_tree)
                    else:
                        node = new_tree.try_pair(v1, v2, new_id=nid, tag=tag)
                        nodes_dict.update({nid:node})

                    delta, theta = old_tree.compute_delta(node, v1, v2)
                    delta = delta.numpy()[0][0]
                    
                    if max_delta is None or (max_delta is not None and delta > max_delta):
                        nid1 = v1
                        nid2 = v2
                        new_node = node
                        new_theta = theta
                        max_delta = delta
                        
                    new_tree.ctrlz(node, v1, v2)
                    tested += 1
                    if tested % STOP == 0:
                        print("       >> tested couples :", tested, "on", num_couples, "in ", dt.now()-start2)
            z += 1
        t += 1

        if len(second_layer) == 1:
            print("len(second_layer) == 1")
            break

        print("       >> best delta     :", max_delta)
        if max_delta <= 0 or new_node is None:
            print("new node is None or max_delta ")
            break

        unbornify(second_layer, nid1.identifier, nid2.identifier, nodes_dict)
        new_tree.parentify(pid=new_node, nid1=nid1, nid2=nid2, theta=new_theta)
        old_tree = new_tree
        new_tree.show()

    print("       >> len(nodes_dict):", len(nodes_dict))
    print("[TIME] -- growing took ", dt.now()-start)
    return new_tree

def retrieve_node(nid, node, nodes_dict, v1, v2, new_tree):
    """
    Retrieves the node from the dictionary and does some magic to make it work
    NOTE: this tree lib really sucks sometimes
    """
    node = nodes_dict[nid]
    node = new_tree.create_node(tag=node.tag,
                                identifier=node.identifier,
                                parent=new_tree.root,
                                g=node.g,
                                alpha=node.alpha,
                                b=node.b,
                                l=node.l,
                                x=node.x,
                                w=node.w,
                                h_val=node.h_val,
                                exph_val=node.exph_val)
    new_tree.move_node(v1.identifier, node.identifier)
    new_tree.move_node(v2.identifier, node.identifier)
    return node

def unbornify(root_children, nid1, nid2, nodes_dict):
    """
    Removes entries in the dictionary that will never be used
    If I merged v1 and v2, all pairs I can compute involving v1 or v2 are now useless and only waste memory
    """
    len1 = len(nodes_dict)
    start = dt.now()
    print("       >> len(nodes_dict):", len1)
    for v in root_children:
        if (nid1 != v.identifier) and (nid2 != v.identifier):
            id_tocheck = IDentify(v.identifier, nid1)
            if id_tocheck in nodes_dict:
                nodes_dict.pop(id_tocheck)
            id_tocheck = IDentify(v.identifier, nid2)
            if id_tocheck in nodes_dict:
                nodes_dict.pop(id_tocheck)
    print("       >> unbornify removed", len1-len(nodes_dict), "entries in ", dt.now()-start)
            

######### SAVE & LOAD ###########


def str_to_tensor(str_val, dtype="float32"):
    """
    Converts np.array string to tf.Tensor
    """
    stripped = []
    list_val = str_val.strip('[]\n').split()
    for elem in list_val:
        e = elem.strip("[]")
        if e != "" and e != "]":
            stripped.append(e)

    return tf.convert_to_tensor(np.array(stripped, dtype=dtype))


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
                    x=data["x"] if isinstance(data["b"],int) else str_to_tensor(data["x"]),
                    w=data["w"] if (isinstance(data["w"],int) or isinstance(data["w"],float)) else str_to_tensor(data["w"]),
                    l=data["l"],
                    exph_val=data["exph"] if "exph" in data else None)

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
    res_tree.A = dict_tree["tree_data"]["A"]
    __parse_json_tree(res_tree, dict_tree, parent=None)

    res_tree.show()
    return res_tree


def txt_log(tree, start_time, path="./log.txt"):
    '''with open(path, "a") as f:
        f.write("####################################################")
        f.write("Time elapsed: {}\n".format(dt.now() - start_time))
        #f.write(tree.show())
        f.write("--------------------------------")
        #f.write(tree.info())
    f.close()'''
    raise NotImplementedError


