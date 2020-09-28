import os
import json
import numpy as np
import random as rd
import treelib as tl
import tensorflow as tf

from math import sqrt, log
from datetime import datetime as dt

POSITIVE_IMAGE_SET = "./dataset/train_val/"
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
        self.w = None
        self.beta = 1
        self.b = b
        self.l = l          # initial lambda
        self.x = x
        
    def compute_h(self, x):
        """
        Compute the node's hypotesis on x
        """
        return tf.matmul(tf.transpose(self.w), x) + self.b

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

    def __init__(self, tree=None, deep=False, node_class=InterpretableNode, identifier=None, gamma=1, s=None):
        self.gamma = gamma
        self.s = s
        super(InterpretableTree, self).__init__(tree=tree, deep=deep,node_class=InterpretableNode, identifier=identifier)

    
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
        auxtree = InterpretableTree(self.subtree(self.root), deep=True, gamma=self.gamma, s=self.s)
        
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
        It also normalizes g and b and computes gamma
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
        print("[TIME] -- vectorify took         ", dt.now()-start)


    #####################################################################################

    # OVERRIDE #
    def to_dict(self, nid=None, key=None, sort=True, reverse=False, with_data=False):
        """
        Transform the whole tree into a dict, saving also node parameters.
        """
        nid = self.root if (nid is None) else nid
        ntag = self[nid].tag
        tree_dict = {ntag: {"children": []}}
        data = {
            "alpha" : str(self[nid].alpha.numpy()),
            "g"     : str(self[nid].g.numpy()),
            "b"     : str(self[nid].b.numpy()) if not isinstance(self[nid].b,int) else self[nid].b,
            "l"     : self[nid].l
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
    def to_json(self, sort=True, reverse=False):
        """
        To format the tree in JSON format.
        """
        return json.dumps(self.to_dict(with_data=True, sort=sort, reverse=reverse))


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
    def str_to_tensor(self, str_val):
        """
        Converts string np.array to tf.Tensor
        """
        np_val = np.array(str_val.strip('[]\n').split(), dtype="float32")
        return tf.convert_to_tensor(np_val)


    @classmethod
    def __parse_json_tree(self, tree, current, parent=None):
        """
        Parse a tree from a JSON object returned by from_json()
            - tree      : tree instance where the parsed json_tree will be saved 
            - current   : node to parse, initially the JSON tree returned by from_json() 
            - parent    : parent node of current, initially None
        """
        par_tag = list(parent.keys())[0] if parent is not None else parent
        curr_tag = current if isinstance(current,str) else list(current.keys())[0]
        # print("<On node ->", curr_tag)

        data = current[curr_tag]["data"]
        tree.create_node(tag=curr_tag, identifier=curr_tag, parent=par_tag, 
                        alpha=self.str_to_tensor(data["alpha"]),
                        g=self.str_to_tensor(data["g"]), 
                        b=data["b"] if isinstance(data["b"],int) else self.str_to_tensor(data["b"]), 
                        l=data["l"])
        if "children" not in current[curr_tag].keys(): #isinstance(current,str):
            # print(" | -- on leaf ", curr_tag)
            return 

        else:
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


#########################################################################################
def e_func(p, q):
    return log(rd.randint(1, 5))
