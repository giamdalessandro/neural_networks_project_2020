import numpy as np
import tensorflow as tf
import treelib as tl
from math import sqrt

NUM_FILTERS = 512
LAMBDA_0 = 0.000001

class DecisionNode(tl.Node):
    """
    Class for the node of a decision tree
        - alpha = boolean vector to determine which weights are used
        - g     = weights of the node (?)
        - w     = alpha * g
        - b     = 0 (???)
    Parameters:
        - l = 10^-6 --- lambda = 10^-6 * sqrt(||omega_node||)
        - beta  = 1
        - image = path to the image that generated the node
    """

    def __init__(self,
                 tag=None,
                 identifier=None,
                 parent=None,
                 data=None,
                 b=0, l=LAMBDA_0,
                 alpha=np.ones(shape=(NUM_FILTERS)),
                 g=np.zeros(shape=(NUM_FILTERS)),
                 image=None):
        super().__init__(tag=tag, identifier=identifier)
        self.alpha = tf.convert_to_tensor(alpha)
        self.g = tf.convert_to_tensor(g)
        self.w = tf.math.multiply(self.alpha, self.g)
        self.beta = 1
        self.b = b
        self.l = l          # initial lambda
        self.image = image  # path to the image that generated the node if the node is leaf, else is None
                            # and the images need to be searched in all children nodes

    def compute_h(self, x):
        """
        Compute the node's hypotesis on x
        """
        return tf.matmul(tf.transpose(self.w), x) + self.b

    def print_info(self):
        print("[NODE] -- tag:   ", self.tag)
        #print("       -- alpha: ", self.alpha.shape)
        #print("       -- g:     ", self.g.shape)
        #print("       -- w:     ", self.w.shape)
        #print("       -- b:     ", self.b)
        print("       -- lamba: ", self.l)
        if self.is_leaf():
            print("       -- leaf of image : ", self.image)
        elif self.is_root():
            print("       -- root")
        else:
            print("       -- generic node")
        print("------------------------------")



#####################################################################################################################

class DecisionTree(tl.Tree):
    """
    Class for the decision tree
        - gamma = (E[y_i])^-1, parameter computed on the set of all positive images 
    """
    def __init__(self, gamma):
        super().__init__(node_class=DecisionNode)
        self.gamma = gamma

    # OVERRIDE
    def create_node(self, tag=None, identifier=None, parent=None, g=np.zeros(shape=(NUM_FILTERS)),
                    alpha=np.ones(shape=(NUM_FILTERS)), b=0, image=None, l=LAMBDA_0):
        """
        Create a child node for given @parent node. If ``identifier`` is absent,
        a UUID will be generated automatically.
        """
        node = self.node_class(tag=tag, identifier=identifier,
                               data=None, g=g, alpha=alpha, b=b, l=l, image=image)
        self.add_node(node, parent)
        return node

    def find_gab(self, n1, n2):
        """
        Finds g, alpha and b optimal for the new node
        """
        #raise NotImplementedError
        return np.ones(shape=(NUM_FILTERS)), np.ones(shape=(NUM_FILTERS)), 5

    def merge(self, n1, n2):
        """
        Merges nodes n1 and n2 to create a parent n, to whom n1 and n2 become children 
        """
        g,a,b = self.find_gab(n1, n2)
        l = LAMBDA_0*sqrt(len(self.leaves(n1)) + len(self.leaves(n2)))
        tag = n1 + n2
        node = self.create_node(tag=tag, identifier=None, parent='root',
                         alpha=a, g=g, b=b, l=l, image=None)
        self.move_node(n1, node.identifier)
        self.move_node(n2, node.identifier)
        return node
    
    def grow(self):
        """
        Grows the tree merging nodes until the condition is met
        """
        raise NotImplementedError


t = DecisionTree(gamma=2)
root = t.create_node(tag="root", identifier='root')
n1 = t.create_node(tag="n1", identifier='n1', parent='root')
n2 = t.create_node(tag="n2", identifier='n2', parent='root')
n3 = t.create_node(tag="n3", identifier='n3', parent='root')
n = t.merge('n1','n2')
m = t.merge(n1.identifier, n.identifier)

print("\n\n")
t.show()
root.print_info()
n1.print_info()
n2.print_info()
n.print_info()
