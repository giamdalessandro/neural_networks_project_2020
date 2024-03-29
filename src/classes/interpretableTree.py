import os
import json
import fnmatch
import numpy as np
import random as rd
import treelib as tl
import tensorflow as tf

from classes.interpretableNode import InterpretableNode
from classes.tree_utils import *
from utils.A_utils import compute_A 


from math import sqrt, log, exp
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


class InterpretableTree(tl.Tree):
    """
    Class for the decision tree
        - gamma = (E[y_i])^-1, parameter computed on the set of all positive images 
    """

    def __init__(self,
                 E=0,
                 s=None,
                 theta=0,
                 gamma=0,
                 tree=None,
                 deep=False,
                 identifier=None, 
                 node_class=InterpretableNode):
        self.s = s
        self.E = E
        self.theta = theta
        self.gamma = gamma
        self.A = None
        super(InterpretableTree, self).__init__(tree=tree, deep=deep, node_class=InterpretableNode, identifier=identifier)

    # OVERLOAD #
    def _clone(self, identifier=None, with_tree=False, deep=False):
        """
        To overload for custom classes to avoid rewriting subtree() and remove_subtree()
        """
        return InterpretableTree(tree=self if with_tree else None, deep=deep, identifier=identifier, gamma=self.gamma, s=self.s)

    # OVERRIDE #
    def to_dict(self, nid=None, key=None, sort=True, reverse=False, with_data=False):
        """
        Transform the whole tree into a dict, saving also node parameters.
        """
        nid = self.root if (nid is None) else nid
        ntag = self[nid].tag
        tree_dict = {ntag: {"children": []}}
        data = {
            "g": sanitize((self[nid].g)),
            "b": sanitize(self[nid].b),
            "w": sanitize(self[nid].w),
            "x": sanitize(self[nid].x),
            "l": sanitize(self[nid].l),
            "exph_val": sanitize(self[nid].exph_val),
            "alpha": sanitize(self[nid].alpha)
        }
        if with_data:
            tree_dict[ntag]["data"] = data

        if self[nid].expanded:
            queue = [self[i] for i in self[nid].successors(self._identifier)]
            key = (lambda x: x) if (key is None) else key
            if sort:
                queue.sort(key=key, reverse=reverse)

            for elem in queue:
                tree_dict[ntag]["children"].append(
                    self.to_dict(elem.identifier, with_data=with_data, sort=sort, reverse=reverse))
            if len(tree_dict[ntag]["children"]) == 0:
                tree_dict = self[nid].tag if not with_data else \
                    {ntag: {"data": data}}
            return tree_dict

    # OVERRIDE #
    def create_node(self, tag=None, identifier=None, parent=None, g=np.zeros(shape=(NUM_FILTERS)),
                    alpha=np.ones(shape=(NUM_FILTERS)), b=0, l=LAMBDA_0, x=None, y=0, w=None, h_val=None, exph_val=None):
        """
        Create a child node for given @parent node. If ``identifier`` is absent,
        a UUID will be generated automatically.
        """
        node = self.node_class(tag=tag, identifier=identifier,
                               data=None, g=g, alpha=alpha, b=b, l=l, x=x, y=y, w=w, h_val=h_val, exph_val=exph_val)
        self.add_node(node, parent)
        return node

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
        print("       -- depth:...........", self.depth())
        for i in range(self.depth()):
            l = 0
            if i == 7 or i == 8:
                l = len(self.leaves())
            else:
                for node in self.all_nodes_itr():
                    if self.level(node.identifier) == i:
                        l+=1
            print("       -- level", i, "has", l, "nodes")
        #print("       -- gamma:...........", self.gamma if isinstance(self.gamma, float) else self.gamma.numpy())
        #print("       -- s (shape):.......", self.s.shape)
        print("-------------------------------------------------")

    def save2json(self, save_name, save_folder="./forest"):
        """
        Saves a tree to a JSON file
            - save_name  : save file name (w/o '.json')
            - save_folder: folder where to save JSON trees
        """
        tree_data = {
            "E"     : str(self.E)               if isinstance(self.E, int) or isinstance(self.E, float) else str(self.E.numpy()),
            "s"     : str(self.s.numpy())       if tf.is_tensor(self.s) else str(self.s),
            "gamma" : str(self.gamma.numpy())   if tf.is_tensor(self.gamma) else self.gamma,
            "A"     : str([list(self.A[d]) for d in range(len(self.A))])   if self.A is not None else self.A,
            "theta" : str(self.theta)
        }
        json_tree = json.loads(self.to_json(with_data=True))
        json_tree.update({"tree_data" : tree_data})

        file_path = os.path.join(save_folder, save_name + ".json")
        with open(file_path, "w") as f:
            json.dump(json_tree, f, indent=2)
            f.close()

        print("Tree saved in ", file_path)
        return file_path

    def init_leaves(self, trained_model, pos_image_folder):
        """
        Initializes a root's child for every image in the positive image folder and returns the list of all predictions done
            >> find . -type f -print0 | xargs -0 mv -t .
            # command to copy all filesf from subdirectories of the current directory in the current directory
        """
        i = 0
        y_dict = {}
        s_list = []
        start = dt.now()

        root = self.create_node(tag="root", identifier='root')

        flat_model = Model(inputs=trained_model.input, outputs=trained_model.get_layer("flatten").output)
        fc3_model = Model(inputs=trained_model.input, outputs=trained_model.get_layer("fc3").output)

        for img in os.listdir(pos_image_folder):
            if img.endswith('.jpg') and img[0] == '2':
                test_image = load_test_image(folder=pos_image_folder, fileid=img)
                flat_output = flat_model.predict(test_image)
                # check if the flat output contains nan values
                if 'nan' in str(flat_output):
                    print("[WARN] -- nan value found in", img)
                    continue

                # we take only the positive prediction score
                fc3_output = fc3_model.predict(test_image)[0][0]
                y = trained_model.predict(test_image)[0][0]          # after softmax
                y_dict.update({img[:-4]: fc3_output})

                g = compute_g(trained_model, flat_output)
                x = tf.reshape(flat_output, shape=(7, 7, 512))
                
                # inner product between g and x
                b = tf.subtract(fc3_output, tf.reduce_sum(tf.math.multiply(g, x), axis=None))

                s = tf.math.reduce_mean(x, axis=[0, 1])
                s_list.append(s)
                self.create_node(tag=img[:-4], identifier=img[:-4],
                                parent='root', g=g, alpha=tf.ones(shape=512), b=b, x=x, y=y)
                i += 1
                print(">> created", i, "nodes --", img)
                if i==STOP:
                    break
                # TEST IF g and b ARE ACCURATE ENOUGH - IS WORKING! #
                # print("\nORIGINAL y -- CALULATED y")
                # print(fc3_output, " = ", tf.add(tf.reduce_sum(tf.math.multiply(g, x), axis=None), b).numpy())

        # sum on s to obtain s(1,1,512) / # images in the positive set
        self.s = tf.reduce_sum(s_list, axis=0)/len(self.all_nodes())
        
        print("[TIME] ----- init leaves took        ", dt.now()-start)
        return y_dict

    def vectorify(self, y_dict):
        """
        Forall leaf in self:
            - vectorifies x and g (using the prev computed s)
            - normalizes g and b 
            - updates w = g°alpha = g
            - computes gamma
        """
        gamma = 0
        x_dict = {}
        start = dt.now()

        for node in self.leaves():
            # computation of x and g
            node.g = tf.multiply(tf.math.scalar_mul(1/L, self.s), vectorify_on_depth(node.g))
            node.x = tf.divide(vectorify_on_depth(node.x), self.s)
            node.x = tf.reshape(node.x, shape=(512, 1))     # reshape in order to use mat_mul in vectorify
            # normalization of g and b
            norm_g = tf.norm(node.g, ord=2)
            node.b = tf.divide(node.b, norm_g)
            node.g = tf.divide(node.g, norm_g)
            # computation of w
            if node.g.shape != [512, 1]:
                node.g = tf.reshape(node.g, shape=(512, 1))
            node.w = node.g
            # computation of gamma using normalized y_i
            gamma += y_dict[node.tag]/norm_g

        cardinality = len(self.leaves())
        self.gamma = cardinality/gamma              # gamma viene usata solo per calcolare E
        print("[TIME] ----- vectorifing leaves took ", dt.now()-start)
        return x_dict

    def init_E(self):
        """
        Only to be used the first time
        """
        E = 0
        theta = 0
        start = dt.now()
        for node in self.leaves():
            node.h_val, node.exph_val = node.exph(self.gamma)
            E += node.h_val
            theta += node.exph_val

        self.E = E              # - len(self.leaves())*log(self.theta)
        self.theta = theta
        print("[TIME] ----- computing E took        ", dt.now()-start)

    def compute_delta(self, node, v1, v2):
        """
        computes delta log E using pre existent values in self
        NOTE: this computes the delta between E_t+1 and E_t; current self has E_t, theta_t
        delta and new theta is returned
        """
        new_theta = (self.theta + node.exph_val - v1.exph_val - v2.exph_val)
        a = node.h_val - v1.h_val - v2.h_val
        b = len(self.leaves()) * log(self.theta / new_theta)
        return (1 + self.gamma*a + b), new_theta

    def update_node_values(self, n1, n2):     # SEMI FAKE #
        """
        Finds g, alpha and b optimal for the new node
        Computes also w and l
        """
        #start = dt.now()

        b = 0
        g = optimize_g(n1.g, n2.g)
        
        Xs = []
        Ys = []
        if n1.is_leaf():
            Xs.append(tf.reshape(tf.multiply(g, n1.x), shape=[512]))
            Ys.append(n1.y)
        else:
            for leaf in self.leaves(nid=n1.identifier):
                Xs.append(tf.reshape(tf.multiply(g, leaf.x), shape=[512]))
                Ys.append(leaf.y)

        if n2.is_leaf():
            Xs.append(tf.reshape(tf.multiply(g, n1.x), shape=[512]))
            Ys.append(n1.y)
        else:
            for leaf in self.leaves(nid=n2.identifier):
                Xs.append(tf.reshape(tf.multiply(g, leaf.x), shape=[512]))
                Ys.append(leaf.y)
        alpha = optimize_alpha(Xs, Ys)

        w = tf.math.multiply(alpha, g)
        l = LAMBDA_0 * sqrt(len(self.leaves(n1.identifier)) +
                            len(self.leaves(n2.identifier)))
        #print("       >> optimizing params took: ", dt.now()-start)
        return g, alpha, b, w, l

    def try_pair(self, nid1, nid2, new_id, tag):
        """
        Merges nodes nid1 and nid2 to create a parent n, to whom nid1 and nid2 become children
        The new node will have exp(h) = exp(h_nid1) + exp(h_nid2)
        """
        g, alpha, b, w, l = self.update_node_values(nid1, nid2)
        
        node = self.create_node(tag=tag, parent='root', alpha=alpha, g=g, b=b, l=l, x=None, w=w, identifier=new_id)
        
        # calculates the value of the new node's exph to use it later
        node.h_val    = 0
        node.exph_val = 0

        if nid1.is_leaf():
            aux = node.exph(self.gamma, nid1.x)
            node.h_val += aux[0]
            node.exph_val += aux[1]
        else:
            node.h_val += nid1.h_val
            node.exph_val += nid1.exph_val

        if nid2.is_leaf():
            aux = node.exph(self.gamma, nid2.x)
            node.h_val += aux[0]
            node.exph_val += aux[1]
        else:
            node.h_val += nid2.h_val
            node.exph_val += nid2.exph_val

        self.move_node(nid1.identifier, node.identifier)
        self.move_node(nid2.identifier, node.identifier)
        return node
  
    def ctrlz(self, pid, nid1, nid2):
        """
        Undoes what try_pair() does
        """
        self.move_node(nid1.identifier, self.root)
        self.move_node(nid2.identifier, self.root)
        if self.contains(pid.identifier):
            self.remove_node(pid.identifier)
        else:
            print("[[[ERR]]]: i tried to delete this node: <tag:", pid.tag, ", id:", pid.identifier, "> but it was not in the tree! ur such a git dumbass")
            print("let's see the tree")
            self.show()

    def parentify(self, pid, nid1, nid2, theta):
        """
        Gives hope to two orphan children
        """
        self.theta = theta
        self.add_node(pid, parent=self.root)
        self.move_node(nid1.identifier, pid.identifier)
        self.move_node(nid2.identifier, pid.identifier)


    ########## METRICS ###########

    def cos_similarity(self, x, y):
        """
        Computes cosine similarity between x and y tensors
        """
        norm_x = tf.nn.l2_normalize(x,0)        
        norm_y = tf.nn.l2_normalize(y,0)
        return tf.reduce_sum(tf.multiply(norm_x,norm_y))

    def predict(self, g, x, level=1):
        # only used to compute g, no need for activation
        if level == 1:
            children = self.children(self.root)
        else:
            children = self.get_generation(level)
        node = self.find_best_node(children, g)        
        return tf.matmul(tf.reshape(node.w, shape=(512, 1)), x, transpose_a=True).numpy()/1000


    def get_generation(self, level):
        """
        Gets all nodes at a certain level of the tree
        """
        if level > 0:
            return [node for node in self.all_nodes_itr() if self.level(node.identifier) == level]
        else:
            return self.leaves()
    

    def find_best_node(self, nodes, g):
        max_similarity = self.cos_similarity(g, nodes[0].w)
        best_node = nodes[0]
        for n in nodes:
            similarity = self.cos_similarity(g, n.w)
            if similarity > max_similarity:
                max_similarity = similarity
                best_node = n
        return best_node

    def compute_hatrho(self, x, g, t, level):
        """
        Returns hatrho for the jaccard similarity
        """
        nodes = self.children(self.root) if level == 1 else self.get_generation(level)
        bestnode = self.find_best_node(nodes, g) 
        rho = tf.multiply(bestnode.w,x)
        return tf.maximum(0, tf.multiply(rho, tf.math.sign(t)))


    def compute_g_strano(self, x, g, level=1):
        nodes = self.children(self.root) if level == 1 else self.get_generation(level)        
        bestnode = self.find_best_node(nodes, g)
        rho = tf.multiply(tf.reshape(bestnode.w, shape=(512,1)), x)
        g_outo = tf.matmul(self.A, rho, transpose_a=True)
        return tf.multiply(1/tf.reduce_sum(g_outo), g_outo)
