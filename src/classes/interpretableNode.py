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

from classes.tree_utils import L, DTYPE, LAMBDA_0, NUM_FILTERS


class InterpretableNode(tl.Node):
    """
    Class for the node of a decision tree
        - alpha = boolean vector to determine which weights are used
        - g     = weights of the node (?)
        - w     = alpha * g
        - b     = 0 (???)
    Parameters:
        - l = 10^-6 --- lambda = 10^-6 * sqrt(||omega_node||)
        - image = path to the image that generated the node
    """

    def __init__(self,
                 x=0,
                 b=0,
                 y=0,
                 w=None,
                 tag=None,
                 data=None,
                 l=LAMBDA_0,
                 h_val=None,
                 parent=None,
                 exph_val=None,
                 identifier=None,
                 g=np.zeros(shape=(NUM_FILTERS)),
                 alpha=np.ones(shape=(NUM_FILTERS))):
                 
        super().__init__(tag=tag, identifier=identifier)
        
        # if root, self.b stores 's'
        self.b = b
        
        # if root, self.l stores 'gamma'
        self.l = l
        
        self.x = x
        self.g = g if tf.is_tensor(g) else tf.convert_to_tensor(g)
        self.alpha = alpha if tf.is_tensor(alpha) else tf.convert_to_tensor(alpha)
        self.y = y      # prediction AFTER softmax
        self.w = w
        self.h_val = h_val
        self.exph_val = exph_val


    def h(self, xx=None):
        """
        Compute the node's hypotesis on x 
        """
        x = self.x if xx is None else xx

        w = self.w if self.w.shape == [512, 1] else tf.reshape(self.w, shape=[512, 1])
        x = tf.reshape(x, shape=[512, 1]) if (x is not None and x.shape != [512, 1]) else x

        if w is None:
            print("[ERR] >> w is None ------------------------------------------------")
        if x is None:
            print("[ERR] >> x is None ------------------------------------------------")
        if self.b is None:
            print("[ERR] >> b is None ------------------------------------------------")
        
        return tf.matmul(w, x, transpose_a=True) + self.b

    def exph(self, gamma, x=None):
        """
        Compute the exp of the node's hypotesis on x: e^[gamma * h(x)] = hh(x)
        Returns:
            - #0    h(x)
            - #1    hh(x)
        """
        h = self.h(x)
        return h, exp(gamma * h)


    def print_info(self):
        if self.is_root():
            print("[ROOT] -- root")
            print("       -- s:     ", self.b.shape)
            print("       -- gamma: ", self.l)
        
        else:
            if self.is_leaf():
                print("[LEAF] -- tag:   ", self.tag)
            else:
                print("[NODE] -- tag:   ", self.tag)

            print("       -- alpha: ", self.alpha.shape)
            print("       -- g:     ", self.g.shape,
                  " ||g|| = ", tf.norm(self.g, ord=2).numpy())
            print("       -- x:     ", self.x.shape)
            print("       -- w:     ", self.w.shape if self.w is not None else self.w)
            print("       -- b:     ", self.b.numpy())
            print("       -- lamba: ", self.l)
        print("------------------------------")
