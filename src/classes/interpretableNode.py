from classes.tree_utils import *

class InterpretableNode(tl.Node):
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
                 b=0, l=LAMBDA_0, x=0,
                 alpha=np.ones(shape=(NUM_FILTERS)),
                 g=np.zeros(shape=(NUM_FILTERS))):
        super().__init__(tag=tag, identifier=identifier)

        self.alpha = alpha if tf.is_tensor(
            alpha) else tf.convert_to_tensor(alpha)
        self.g = g if tf.is_tensor(g) else tf.convert_to_tensor(g)
        if g.shape == [512, ]:
            self.w = tf.math.multiply(alpha, g)
        else:
            self.w = None
        self.beta = 1
        self.b = b
        self.l = l          # initial lambda
        self.x = x


    def compute_h(self, x):
        """
        Compute the node's hypotesis on x
        """
        return tf.matmul(tf.transpose(self.w), tf.reshape(x, shape=(512, 1)) + self.b)


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
                  "  ||g|| = ", tf.norm(self.g, ord=2).numpy())
            print("       -- x:     ", self.x.shape)
            print("       -- w:     ",
                  self.w.shape if self.w is not None else self.w)
            print("       -- b:     ", self.b.numpy())
            print("       -- lamba: ", self.l)
        print("------------------------------")
