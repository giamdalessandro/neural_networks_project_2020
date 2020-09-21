import tensorflow as tf
import numpy as np
from treelib import Tree, Node


class DecisionNode(Node):
    """
    Class for the node of a decision tree
        - alpha = boolean vector to determine which weights are used
    """
    def __init__(self, alpha=np.zeros(shape=(512))):
        super(DecisionNode, self).__init__()
        self.alpha = alpha
