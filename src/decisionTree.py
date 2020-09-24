import numpy as np
import random as rd
import treelib as tl
import tensorflow as tf

from math import sqrt, log

POSITIVE_IMAGE_SET = "./dataset/train_val/bird"
NUM_FILTERS = 512
LAMBDA_0 = 0.000001
DTYPE = tf.int32
L = 14*14

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
        if tf.is_tensor(alpha):
            self.alpha = alpha
        else: 
            self.alpha = tf.convert_to_tensor(alpha)
        if tf.is_tensor(g):
            self.g = g
        else:
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

##############################################

class DecisionTree(tl.Tree):
    """
    Class for the decision tree
        - gamma = (E[y_i])^-1, parameter computed on the set of all positive images 
    """

    def __init__(self, tree=None, deep=False, node_class=DecisionNode, identifier=None, gamma=1, s=None):
        self.gamma = gamma
        self.s = s
        super(DecisionTree, self).__init__(tree=tree, deep=deep,node_class=DecisionNode, identifier=identifier)


    # OVERRIDE #
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


    # OVERLOAD #
    def _clone(self, identifier=None, with_tree=False, deep=False):
        """
        To overload for custom classes to avoid rewriting subtree() and remove_subtree()
        """
        return DecisionTree(tree=self if with_tree else None, deep=deep, identifier=identifier, gamma=self.gamma, s=self.s)

    
    def find_gab(self, n1, n2):
        """
        Finds g, alpha and b optimal for the new node
        """
        g = tf.random.uniform(shape=[NUM_FILTERS], minval=1, maxval=5, dtype=DTYPE)
        alpha = tf.random.uniform(shape=[NUM_FILTERS], minval=1, maxval=5, dtype=DTYPE)
        b = 0
        return g,alpha,b
    
    
    def merge_nodes(self, nid1, nid2):
        """
        Merges nodes nid1 and nid2 to create a parent n, to whom nid1 and nid2 become children 
        """
        g,a,b = self.find_gab(nid1, nid2)
        l = LAMBDA_0*sqrt(len(self.leaves(nid1)) + len(self.leaves(nid2)))
        tag = nid1 + nid2
        node = self.create_node(tag=tag, identifier=tag, parent='root',
                         alpha=a, g=g, b=b, l=l, image=None)
        self.move_node(nid1, node.identifier)
        self.move_node(nid2, node.identifier)
        return node
    
    
    def try_merge(self, nid1, nid2):
        """
        Returns a new tree with nid1 and nid2 merged
        """
        new_tree = DecisionTree(self.subtree(self.root), deep=True)       # returns a deep copy of the current tree
        new_tree.merge_nodes(nid1, nid2)                     # merges the nodes in the new tree
        return new_tree


    def vectorify(self, x, g):
        """
        Takes x and g tensors and returns two vectors based on the following rules
            - xx[d] = sum(x[h,w,d])/s_d
            - gg[d] = s_d/L^2 * sum(dy/dx[h,w,d])
        """
        xx = tf.divide(self.__depth_vectorify(x), self.s)
        ss = tf.math.scalar_mul(1/L, self.s)
        gg = tf.multiply(ss, g)                                 ### ???
        return xx


    def __depth_vectorify(self, x):
        """
        xx = tf.ones(shape=(2,2,5))
        x = tf.reduce_sum(xx, axis=[0,1])
        # sum over h and w --> outputs a vector of lenght d=5
        """
        return tf.reduce_sum(x, axis=[0, 1])


##############################################

def e_func(p, q):
    return rd.randint(0, 5)


def choose_pair(curr_tree, tree_0):
    """
    Chooses the pair that creates a new tree P s.t. maximizes E(P,Q)-E(Q,Q) with Q being the tree at step 0
    """
    curr_max = 0
    new_tree = None                     # return value
    e_0 = e_func(tree_0, tree_0)

    # set of all second layer's node
    leaves_set = curr_tree.children(curr_tree.root)
    for (v1, v2) in leaves_set:
        
        aux_tree = curr_tree.try_merge(v1, v2)  # returns a tree with v1 and v2 merged
        e = e_func(aux_tree, tree_0)
        
        if log(e - e_0) >= curr_max:
            curr_max = e
            new_tree = aux_tree
    
    return new_tree


def grow(tree_0):
    """
    Grows the tree merging nodes until the condition is met
    """
    curr_tree = tree_0
    e_0 = e_func(tree_0, tree_0)
    while True:
        curr_tree = choose_pair(curr_tree, tree_0)
        e = e_func(curr_tree, tree_0)
        if log(e - e_0) <= 0:
            return curr_tree


def initialize_leaves(model, tree, pos_image_folder=POSITIVE_IMAGE_SET):
        """
        Initializes a root's child for every image in the positive image folder 
        """
        # s = None
        # gamma = 0
        # for image in imagepath
        #       y = predict - softmax
        #       compute g = ...
        #       compute b = y - g°x
        #       compute s_i
        #       s.append(s_i)
        #       compute gamma += y
        #       add leaf with g, alpha=1, b
        #   
        # sum on s to obtain s(1,1,512)
        # gamma = (# of images)/gamme


''' HOW TO OBTAIN s
POSITIVE_IMAGE_SET = "./dataset/train_val/bird"
l = []  # len(POSITIVE_IMAGE_SET) x 512
for i in POSITIVE_IMAGE_SET:
    imported.predict(load_image())
    val = vectorify_on_depth(x)
    val = val / 14*14 (?????????)
    l.append(val)
val = avg(list, axis=0)
val = val / len(list)
'''


''' TEST vectorify on depth 
x = tf.random.uniform(shape=[2, 2, 3], minval=1,
                      maxval=5, dtype=tf.int32, seed=42)
print(x)
print(vectorify_on_depth(x))
'''

### TEST ###

t = DecisionTree(gamma=2)
root = t.create_node(tag="root", identifier='root')
n1 = t.create_node(tag="n1", identifier='n1', parent='root')
n2 = t.create_node(tag="n2", identifier='n2', parent='root')
n3 = t.create_node(tag="n3", identifier='n3', parent='root')
print("original tree")
t.show()

n = t.merge_nodes('n1', 'n2')
st = t.subtree(t.root)
m = t.try_merge('n3', n.identifier)
print("original tree after merging n1 and n2")
t.show()
print("new tree after merging n1n2 and n3")
m.show()

# l = t.children(t.root)
# print(l)
# for i in t.all_nodes():
#    i.print_info()
