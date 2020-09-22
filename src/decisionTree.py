import numpy as np
import tensorflow as tf
from treelib import Tree, Node

NUM_FILTERS = 512

class DecisionNode(Node):
    """
    Class for the node of a decision tree
        - alpha = boolean vector to determine which weights are used
        - g     = weights of the node (?)
        - w     = alpha * g
        - b     = 0 (???)
    Parameters:
        - l = 10^-6 --- lambda = 10^-6 * sqrt(||omega_node||)
        - beta  = 1
        - gamma = (E[y_i])^-1 forall i in Omega+ --- CONST forall nodes
        - image = path to the image that generated the node
    """

    def __init__(self, alpha=np.ones(shape=(NUM_FILTERS)), g=np.zeros(shape=(NUM_FILTERS)), b=0, image=None):
        super(DecisionNode, self).__init__()
        self.alpha = alpha
        self.g = g
        self.w = np.zeros(shape=(NUM_FILTERS))
        self.beta = 1
        self.b = b
        self.l = 0          # lambda
        self.gamma = 0
        self.image = image  # path to the image that generated the node if the node is leaf, else is None
                            # and the images need to be searched in all children nodes

    def compute_h(self, x):
        """
        Compute the node's hypotesis on x
        """
        self.w = tf.math.scalar_mul(self.alpha, self.g)
        return tf.math.multiply(tf.transpose(self.w), x) + self.b

    def print_info(self):
        print("[NODE] -- alpha: ", self.alpha)
        print("[NODE] -- g:     ", self.g)
        print("[NODE] -- w:     ", self.w)
        print("[NODE] -- b:     ", self.b)
        print("[NODE] -- lamba: ", self.l)
        if self.is_leaf():
            print("[NODE] -- leaf of image ", self.image)
        elif self.is_root():
            print("[NODE] -- root")
        else:
            print("[NODE] -- generic node")


#####################################################################################################################

class DecisionTree(Tree):
    """
    Class for the decision tree
        - gamma = (E[y_i])^-1, parameter computed on the set of all positive images
    """
    def __init__(self, gamma):
        super(DecisionTree, self).__init__(node_class=DecisionNode)
        self.gamma = gamma

    def 
