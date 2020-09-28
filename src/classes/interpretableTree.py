import os
import json
import numpy as np
import random as rd
import treelib as tl
import tensorflow as tf

from math import sqrt, log, exp
from datetime import datetime as dt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


NEG_IMAGE_SET_TEST = "./dataset/train_val/test/bird/"
POS_IMAGE_SET_TEST = "./dataset/train_val/test/not_bird/"
NUM_FILTERS = 512
LAMBDA_0 = 0.000001
DTYPE = tf.int32
L = 14*14


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
        
        self.alpha = alpha if tf.is_tensor(alpha) else tf.convert_to_tensor(alpha)
        self.g = g if tf.is_tensor(g) else tf.convert_to_tensor(g)  
        if g.shape == [512,]:
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
        return tf.matmul(tf.transpose(self.w), tf.reshape(x, shape=(512,1)) + self.b)

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
            print("       -- g:     ", self.g.shape, "  ||g|| = ", tf.norm(self.g, ord=2).numpy())
            print("       -- x:     ", self.x.shape)
            print("       -- w:     ", self.w.shape if self.w is not None else self.w)
            print("       -- b:     ", self.b.numpy())
            print("       -- lamba: ", self.l)
            

        print("------------------------------")

##############################################

class InterpretableTree(tl.Tree):
    """
    Class for the decision tree
        - gamma = (E[y_i])^-1, parameter computed on the set of all positive images 
    """

    def __init__(self,
                 eta=0,
                 s=None,
                 gamma=1,
                 tree=None,
                 deep=False,
                 fc3_model=None,
                 flat_model=None,
                 identifier=None, 
                 node_class=InterpretableNode):
        self.s = s
        self.eta = eta
        self.gamma = gamma
        self.flat_model = flat_model
        self.fc3_model = fc3_model
        super(InterpretableTree, self).__init__(tree=tree, deep=deep, node_class=InterpretableNode, identifier=identifier)

    
    def info(self):
        """
        Prints useful info
        """
        size = self.size()
        leaves = len(self.leaves())
        print("-------------------------------------------------")
        print("[TREE] -- nodes:...........", size)
        print("       -- generic nodes:...", size - leaves - 1)
        print("       -- leaves:..........", leaves)
        print("       -- gamma:...........", self.gamma.numpy())
        print("       -- s (shape):.......", self.s.shape)
        print("-------------------------------------------------")
        

    # OVERRIDE #
    def create_node(self, tag=None, identifier=None, parent=None, g=np.zeros(shape=(NUM_FILTERS)),
                    alpha=np.ones(shape=(NUM_FILTERS)), b=0, l=LAMBDA_0, x=0):
        """
        Create a child node for given @parent node. If ``identifier`` is absent,
        a UUID will be generated automatically.
        """
        node = self.node_class(tag=tag, identifier=identifier,
                               data=None, g=g, alpha=alpha, b=b, l=l, x=x)
        self.add_node(node, parent)
        return node


    # OVERLOAD #
    def _clone(self, identifier=None, with_tree=False, deep=False):
        """
        To overload for custom classes to avoid rewriting subtree() and remove_subtree()
        """
        return InterpretableTree(tree=self if with_tree else None, deep=deep, identifier=identifier, gamma=self.gamma, s=self.s)

    
    def find_gab(self, n1, n2):
        """
        Finds g, alpha and b optimal for the new node
        """
        g = tf.random.uniform(shape=[NUM_FILTERS], minval=1, maxval=5, dtype=DTYPE)
        alpha = tf.random.uniform(shape=[NUM_FILTERS], minval=1, maxval=5, dtype=DTYPE)
        b = 0
        return g,alpha,b


    def __parentify(self, nid1, nid2, pid):
        """
        Gives hope to two orphan children
        """
        self.add_node(pid, parent=self.root)
        self.move_node(nid1, pid.identifier)
        self.move_node(nid2, pid.identifier)
    

    def __shallow_merge(self, nid1, nid2, tag=None):
        """
        Merges nodes nid1 and nid2 to create a parent n, to whom nid1 and nid2 become children
        """
        g,a,b = self.find_gab(nid1, nid2)
        l = LAMBDA_0 * sqrt(len(self.leaves(nid1)) + len(self.leaves(nid2)))
        tag = nid1 + nid2 if tag is None else tag
        node = self.create_node(tag=tag, parent='root', alpha=a, g=g, b=b, l=l, x=None, identifier=tag)
        self.move_node(nid1, node.identifier)
        self.move_node(nid2, node.identifier)
        return node
        
    
    def __shallow_unmerge(self, nid1, nid2):
        """
        Undoes what __shallow_merge does
        """
        killed = self.parent(nid1)
        self.move_node(nid1, self.root)
        self.move_node(nid2, self.root)
        self.remove_node(killed.identifier)

    
    def choose_pair(self, tree_0, p):
        """
        Copies one time the tree and then executes all merging operation on this new tree
        After a merge operation, it calculates delta E and then reverts back the tree to its previous form
        Note: need to return the max value of E and the two nodes which merged can generate the tree on which we can compute E
        """
        curr_max = 0                    # current max
        e_0 = e_func(tree_0, tree_0)    # E(Q,Q)

        # value to be used later when returning final tree
        n1  = None
        n2  = None
        pid = None

        # copy current tree (just one time)
        auxtree = InterpretableTree(s=self.s,
                                    deep=True,
                                    eta=self.eta,
                                    gamma=self.gamma,
                                    fc3_model=self.fc3_model,
                                    flat_model=self.flat_model,
                                    tree=self.subtree(self.root))

        # set of all second layer's node
        second_layer = self.children(self.root)
        
        z = 1
        it = 1
        for v1 in second_layer:
            if z < len(second_layer):
                for v2 in second_layer[z:]:

                    node = auxtree.__shallow_merge(v1.identifier, v2.identifier, 'p_'+str(p)+'it_'+str(it))
                    e = e_func(auxtree, tree_0)

                    if e-e_0 >= curr_max:       # save node with children for future merge
                        curr_max = e
                        pid = node
                        n1 = v1.identifier
                        n2 = v2.identifier

                    auxtree.__shallow_unmerge(v1.identifier, v2.identifier)
                    it += 1
                    if it%1000 == 0:
                        print("       >>        >> tested couples:", it)
            z += 1
        print("       >> generated couples: ", it-1)
        # merges the chosen nodes and returns the tree (deep copied at the beginning)
        auxtree.__parentify(n1, n2, pid)
        return auxtree


    def __vectorify_on_depth(self, x):
        """
        xx = tf.ones(shape=(2,2,5))
        x = tf.reduce_sum(xx, axis=[0,1])
        # sum over h and w --> outputs a vector of lenght d=5
        """
        return tf.reduce_sum(x, axis=[0, 1])


    def vectorify(self, y_dict):
        """
        Forall leaf in self, vectorifies x and g (using the prev computed s) and updates w = g°x
        It also normalizes g and b and computes gamma and eta
        """
        start = dt.now()
        gamma = 0
        for node in self.leaves():
            node.g = tf.multiply(tf.math.scalar_mul(1/L, self.s), self.__vectorify_on_depth(node.g))  # ???
            node.x = tf.divide(self.__vectorify_on_depth(node.x), self.s)
            # normalization of g and b
            norm_g = tf.norm(node.g, ord=2)
            node.b = tf.divide(node.b, norm_g)
            node.g = tf.divide(node.g, norm_g)
            # computation of w
            node.w = tf.math.multiply(node.alpha, node.g)
            # computation of gamma using normalized y_i
            gamma += y_dict[node.tag]/norm_g
        
        cardinality = len(self.leaves())
        self.gamma = cardinality/gamma              # gamma viene usata solo per calcolare E
        self.__compute_eta()                        # computes eta which is constant for all trees
        print("[TIME] -- vectorify took         ", dt.now()-start)



    def save2json(self, save_name, save_folder="./forest"):
        """
        Saves a tree to a JSON file
            - save_name  : save file name (w/o '.json')
            - save_folder: folder where to save JSON trees
        """
        json_tree = json.loads(self.to_json())

        file_path = os.path.join(save_folder, save_name + ".json")
        with open(file_path, "w") as f:
            json.dump(json_tree, f, indent=2)
            f.close()

        print("Tree saved in ", file_path)
        return file_path            


    @classmethod
    def __parse_json_tree(self, tree, current, parent=None):
        """
        Parse a tree from a JSON object returned by from_json()
            - tree      : tree instance to witch the parsed json_tree will be saved 
            - current   : node to parse, initially the JSON tree returned by from_json() 
            - parent    : parent node of current, initially None
        """
        par_tag = list(parent.keys())[0] if parent is not None else parent
        curr_tag = current if isinstance(current,str) else list(current.keys())[0]
        # print("<On node ->", curr_tag)

        if isinstance(current,str):
            # print(" | -- on leaf ", curr_tag)
            tree.create_node(tag=curr_tag, identifier=curr_tag, parent=par_tag)
            return 

        else:
            tree.create_node(tag=curr_tag, identifier=curr_tag, parent=par_tag)
            for child in current[curr_tag]["children"]:
                # print("-- on child ", child)

                self.__parse_json_tree(tree, current=child, parent=current)
                
        return


    @classmethod
    def from_json(self, save_path):
        """
        Loads a tree from a JSON file
            - save_path : path to the JSON tree file to load
        """
        with open(save_path, "r") as f:
            dict_tree = json.load(f)
            #print(dict_tree)

        res_tree = InterpretableTree()
        self.__parse_json_tree(res_tree, dict_tree, parent=None)

        res_tree.show()
        return res_tree


    def choose_best_node(self, g):
        """
        Chooses the best node that fits with g in the second tree layer
        The best node is chosen following the formula v = argmax(cos(g, w_v))
            - g --> vector of given image
        """
        best = None
        aux  = -5
        for v in self.children(self.root):
            cos_similarity = tf.compat.v1.losses.cosine_distance(g, v.w, axis=0)
            if cos_similarity >= aux:
                best = v
        return best

    
    def compute_g(self, inputs):
        '''
        Computes g = dy/dx, where x is the output of the top conv layer after the mask operation,
        and y is the output of the prediction before the softmax.
            - model:  the pretrained modell on witch g will be computed;
            - imputs: x, the output of the top conv layer after the mask operation.
        '''
        fc_1 = self.fc3_model.get_layer("fc1")
        fc_2 = self.fc3_model.get_layer("fc2")
        fc_3 = self.fc3_model.get_layer("fc3")

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(fc_1.variables)

            y = fc_3(fc_2(fc_1(inputs)))
            gradient = tape.gradient(y, fc_1.variables)

        return tf.reshape(tf.reduce_sum(gradient[0], axis=1), shape=(7, 7, 512))

        

    def compute_product_probability(self):
        """
        Returns the product of all P_t(xi) divided by e^|Vt|
        """
        val = 0
        for img in os.listdir(POS_IMAGE_SET_TEST):
            if img.endswith('.jpg'):
                test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
                xi  = self.flat_model.predict(test_image)
                pro = self.__compute_probability(xi)
                val = val * pro
        return val / exp(len(self.children(self.root)))
    

    def __uguale(self, x1, x2):
        if tf.reduce_sum(tf.subtract(x1,x2)) == 0:
            return True
        else:
            return False


    def __compute_probability(self, xi):
        """
        Computes P(xi)
        """
        p1 = 0
        p2 = 0
        xi =  tf.divide(self.__vectorify_on_depth(xi), self.s)
        for img in os.listdir(POS_IMAGE_SET_TEST):
            if img.endswith('.jpg'):
                test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
                xj = self.flat_model.predict(test_image)
                xj = tf.divide(self.__vectorify_on_depth(xj), self.s)
                gj = self.compute_g(xj)
                gj = tf.multiply(tf.math.scalar_mul(1/L, self.s), self.__vectorify_on_depth(gj))

                ### NOTE: FORSE DOBBIAMO NORMALIZZARE g ???? ###

                node = self.choose_best_node(gj)
                if self.__uguale(xi,xj):
                    p1 = exp(self.gamma * node.compute_h(xi))
                    p2 += p1
                else:
                    p2 += exp(self.gamma * node.compute_h(xj))

        for img in os.listdir(NEG_IMAGE_SET_TEST):
            if img.endswith('.jpg'):
                test_image = load_test_image(folder=NEG_IMAGE_SET_TEST, fileid=img)
                xj = self.flat_model.predict(test_image)
                gj = self.compute_g(xj)
                gj = tf.multiply(tf.math.scalar_mul(1/L, self.s), self.__vectorify_on_depth(gj))
                node = self.choose_best_node(gj)
                p2 += exp(self.gamma * node.compute_h(xj))
        return p1/p2


    def __compute_eta(self):
        """
        Computes eta for the current tree
        NOTE: this is a constant parameter to be calculated on tree0 and then copied on the new trees
        """
        return self.compute_product_probability()**(-1)


    def compute_delta(self):
        """
        Computes the delta between E_t and E_0 (stored in eta)
        """
        return log(self.compute_product_probability() * self.eta)

def e_func(p, q):
    return log(rd.randint(1, 5))


def load_test_image(folder, fileid):
    """
    loads and preprocesses default img specified in 'visualize.py' in variable 'path'
    """
    path = os.path.join(folder, fileid)
    img = load_img(path, target_size=(224, 224))      # test image
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
