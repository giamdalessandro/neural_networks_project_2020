import numpy as np

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

    def __init__(self, alpha=np.zeros(shape=(NUM_FILTERS)), g=np.zeros(shape=(NUM_FILTERS)), b=0, image=None):
        super(DecisionNode, self).__init__()
        self.alpha = alpha
        self.g = g
        self.w = np.zeros(shape=(NUM_FILTERS))
        self.beta = 1
        self.b = b
        self.l = 0
        self.gamma = 0
        self.image = image  # path to the image that generated the node if the node is leaf, else is None
                            # and the images need to be searched in all children nodes

#####################################################################################################################

class DecisionTree(Tree):
    """
    Class for the decision tree
        - gamma = (E[y_i])^-1, parameter computed on the set of all positive images
    """
    def __init__(self, gamma):
        super(DecisionTree, self).__init__(node_class=DecisionNode)
        self.gamma = gamma
