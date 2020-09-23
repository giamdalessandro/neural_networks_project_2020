import numpy as np
import treelib as tl
import tensorflow as tf


def vectorify(x, g, s):
    """
    Takes two tensors and returns two vectors
        - xx[d] = sum(x[h,w,d])/s_d
        - gg[d] = s_d/L^2 * sum(dy/dx[h,w,d])
        - s_d = E_i[ E_(h,w)[ x[h,w,d] ] ]
    """
    raise NotImplementedError

def vectorify_on_depth(x):
    """
    xx = tf.ones(shape=(2,2,5))
    x = tf.reduce_sum(xx, axis=[0,1])
    # sum over h and w --> outputs a vector of lenght d=5
    """
    return tf.reduce_sum(x, axis=[0, 1])


def e_func(p, q):
    raise NotImplementedError

def fake_merge(n1, n2):
    raise NotImplementedError

def choose_pair(curr_tree, tree_0):
    """
    Chooses the pair that creates a new tree P s.t. maximizes E(P,Q)-E(Q,Q) with Q being the tree at step 0
    """
    curr_max = 0
    new_tree = None                     # return value
    e_0 = e_func(tree_0, tree_0)
    leaves_set = curr_tree.children(curr_tree.root)         # set of all second layer's node
    for (v1,v2) in leaves_set:
        aux_tree = fake_merge(v1,v2)    # returns a tree with v1 and v2 merged
        e = e_func(aux_tree, tree_0)
        if e >= curr_max:
            curr_max = e 
            new_tree = aux_tree 
    return new_tree


''' TEST vectorify on depth 
x = tf.random.uniform(shape=[2, 2, 3], minval=1,
                      maxval=5, dtype=tf.int32, seed=42)
print(x)
print(vectorify_on_depth(x))
'''
