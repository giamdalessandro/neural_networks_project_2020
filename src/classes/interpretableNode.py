from classes.tree_utils import *

'''
class InterpretableNodeRoot(tl.Node):
    def __init__(self, s=0, tag=None, gamma=0, identifier=None):
        self.s = s
        self.gamma = gamma
        super().__init__(tag=tag, identifier=identifier)        
    
    def print_info(self):
        print("[ROOT] -- root")
        print("       -- s:     ", self.s.shape)
        print("       -- gamma: ", self.gamma)
        print("------------------------------")
'''

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
                 tag=None,
                 data=None,
                 l=LAMBDA_0,
                 parent=None,
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
        
        self.w = tf.math.multiply(alpha, g) if g.shape == [512, ] else None
        self.exph_val = 0

    def h(self):
        """
        Compute the node's hypotesis on x 
        """
        # return tf.matmul(tf.transpose(self.w), tf.reshape(x, shape=(512, 1)) + self.b)
        return tf.matmul(tf.transpose(self.w), self.x + self.b)

    def exph(self, gamma):
        """
        Compute the exp of the node's hypotesis on x - e^[gamma * h(x)]
        """
        return exp(gamma * self.h())


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
