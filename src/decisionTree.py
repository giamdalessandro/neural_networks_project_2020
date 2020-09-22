import numpy as np
import tensorflow as tf
import treelib 

NUM_FILTERS = 512

class DecisionNode(treelib.Node):
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
                 b=0,
                 alpha=np.ones(shape=(NUM_FILTERS)),
                 g=np.zeros(shape=(NUM_FILTERS)),
                 image=None):
        super().__init__(tag=tag, identifier=identifier)
        self.alpha = tf.convert_to_tensor(alpha)
        self.g = tf.convert_to_tensor(g)
        self.w = tf.math.multiply(self.alpha, self.g)
        self.beta = 1
        self.b = b
        self.l = 0          # lambda
        self.image = image  # path to the image that generated the node if the node is leaf, else is None
                            # and the images need to be searched in all children nodes

    def compute_h(self, x):
        """
        Compute the node's hypotesis on x
        """
        return tf.matmul(tf.transpose(self.w), x) + self.b

    def print_info(self):
        print("[NODE] -- tag:   ", self.tag)
        print("[NODE] -- alpha: ", self.alpha.shape)
        print("[NODE] -- g:     ", self.g.shape)
        print("[NODE] -- w:     ", self.w.shape)
        print("[NODE] -- b:     ", self.b)
        print("[NODE] -- lamba: ", self.l)
        if self.is_leaf():
            print("[NODE] -- leaf of image : ", self.image)
        elif self.is_root():
            print("[NODE] -- root")
        else:
            print("[NODE] -- generic node")


#####################################################################################################################

class DecisionTree(treelib.Tree):
    """
    Class for the decision tree
        - gamma = (E[y_i])^-1, parameter computed on the set of all positive images 
    """
    def __init__(self, gamma):
        super().__init__(node_class=DecisionNode)
        self.gamma = gamma

    # override
    def create_node(self, tag=None, identifier=None, parent=None, g=np.zeros(shape=(NUM_FILTERS)),
                    alpha=np.ones(shape=(NUM_FILTERS)), b=0, image=None):
        """
        Create a child node for given @parent node. If ``identifier`` is absent,
        a UUID will be generated automatically.
        """
        node = self.node_class(tag=tag, identifier=identifier,
                               data=None, g=g, alpha=alpha, b=b, image=image)
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
        node = self.create_node(tag=None, identifier=None, parent='root',
                         alpha=a, g=g, b=b, image=None)
        node.print_info()
        self.move_node(n1, node.identifier)
        self.move_node(n2, node.identifier)
    
    def grow(self):
        """
        Grows the tree merging nodes until the condition is met
        """
        raise NotImplementedError


t = DecisionTree(gamma=2)
print("\n\n")

t.create_node(tag="root", identifier='root')
t.create_node(tag="n1", identifier='n1', parent='root')
t.create_node(tag="n2", identifier='n2', parent='root')

t.show()

t.merge('n1','n2')
t.show()
