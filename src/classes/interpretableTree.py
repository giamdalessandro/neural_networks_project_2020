import classes.tree_utils #import *
from classes.interpretableNode import *


class InterpretableTree(tl.Tree):
    """
    Class for the decision tree
        - gamma = (E[y_i])^-1, parameter computed on the set of all positive images 
    """

    def __init__(self,
                 e=0,
                 s=None,
                 theta=0,
                 gamma=0,
                 tree=None,
                 deep=False,
                 identifier=None, 
                 node_class=InterpretableNode):
        self.s = s
        self.e = e
        self.theta = theta
        self.gamma = gamma
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
            "alpha": str(self[nid].alpha.numpy()),
            "g": str(self[nid].g.numpy()),
            "b": str(self[nid].b.numpy()) if not isinstance(self[nid].b, int) else self[nid].b,
            "l": self[nid].l
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
                    alpha=np.ones(shape=(NUM_FILTERS)), b=0, l=LAMBDA_0, x=0):
        """
        Create a child node for given @parent node. If ``identifier`` is absent,
        a UUID will be generated automatically.
        """
        node = self.node_class(tag=tag, identifier=identifier,
                               data=None, g=g, alpha=alpha, b=b, l=l, x=x)
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
        print("       -- gamma:...........", self.gamma.numpy())
        print("       -- s (shape):.......", self.s.shape)
        print("-------------------------------------------------")
        

    # PRIVATE #
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

    def __compute_probability(self, nodei):  #xi):
        """
        Computes P(xi)
        """
        '''
        p1 = 0
        p2 = 0
        xi = tf.divide(vectorify_on_depth(xi), self.s)
        for img in os.listdir(POS_IMAGE_SET_TEST):
            if img.endswith('.jpg'):
                test_image = load_test_image(
                    folder=POS_IMAGE_SET_TEST, fileid=img)
                xj = self.flat_model.predict(test_image)
                xj = tf.divide(vectorify_on_depth(xj), self.s)
                gj = self.compute_g(xj)
                gj = tf.multiply(tf.math.scalar_mul(
                    1/L, self.s), vectorify_on_depth(gj))

                ### NOTE: FORSE DOBBIAMO NORMALIZZARE g ???? ###

                node = self.choose_best_node(gj)
                if uguale(xi, xj):
                    p1 = exp(self.gamma * node.compute_h(xi))
                    p2 += p1
                else:
                    p2 += exp(self.gamma * node.compute_h(xj))

        for img in os.listdir(NEG_IMAGE_SET_TEST):
            if img.endswith('.jpg'):
                test_image = load_test_image(
                    folder=NEG_IMAGE_SET_TEST, fileid=img)
                xj = self.flat_model.predict(test_image)
                gj = self.compute_g(xj)
                gj = tf.multiply(tf.math.scalar_mul(
                    1/L, self.s), vectorify_on_depth(gj))
                node = self.choose_best_node(gj)
                p2 += exp(self.gamma * node.compute_h(xj))
        return p1/p2
        '''
        p1 = 0
        p2 = 0
        xi = nodei.x
        for img in os.listdir(POS_IMAGE_SET_TEST):  # no sbagliato
            # RABARBARO
            # ciclo sui figli della root
            # se figlio v non foglia, prendi sue foglie e usa il suo h_v con x_i delle foglie di v  


            if img.endswith('.jpg'):
                nodej = self.get_node(img)
                xj = nodej.x
                best_node = self.choose_best_node(gj)
                if uguale(xi, xj):
                    p1 = exp(self.gamma * best_node.compute_h(xi))
                    p2 += p1
                else:
                    p2 += exp(self.gamma * best_node.compute_h(xj))

        # RABARBARO -   DOBBIAMO ITERARE ANCHE SULLE NEGATIVE?
        return p1/p2

    def __compute_eta(self):
        """
        Computes eta for the current tree
        NOTE: this is a constant parameter to be calculated on tree0 and then copied on the new trees
        """
        return self.compute_product_probability()**(-1)


    # METHODS #
    def find_gab(self, n1, n2):     # FAKE FAKE FAKE FAKE FAKE FAKE FAKE FAKE FAKE #
        """
        Finds g, alpha and b optimal for the new node
        """
        g = tf.random.uniform(shape=[NUM_FILTERS],
                              minval=1, maxval=5, dtype=DTYPE)
        alpha = tf.random.uniform(
            shape=[NUM_FILTERS], minval=1, maxval=5, dtype=DTYPE)
        b = 0
        return g, alpha, b


    def choose_pair(self, tree_0, p):
        """
        Copies one time the tree and then executes all merging operation on this new tree
        After a merge operation, it calculates delta E and then reverts back the tree to its previous form
        Note: need to return the max value of E and the two nodes which merged can generate the tree on which we can compute E
        """
        curr_max = 0                    # current max

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
                    e = auxtree.compute_delta()

                    if e >= curr_max:       # save node with children for future merge
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

   

    def compute_product_probability(self):
        """
        Returns the product of all P_t(xi) divided by e^|Vt|
        """
        '''
        val = 0
        for img in os.listdir(POS_IMAGE_SET_TEST):
            if img.endswith('.jpg'):
                test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
                xi  = self.flat_model.predict(test_image)
                pro = self.__compute_probability(xi)
                val = val * pro
        return val / exp(len(self.children(self.root)))
        '''
        val = 0
        for img in os.listdir(POS_IMAGE_SET_TEST):
            if img.endswith('.jpg'):
                node = self.get_node(img)
                pro = self.__compute_probability(node)
                val = val * pro
        return val / exp(len(self.children(self.root)))


    def compute_delta(self):
        """
        Computes the delta between E_t and E_0 (stored in eta)
        """
        return log(self.compute_product_probability() * self.eta)



    #### REWRITE ####

    def save2json(self, save_name, save_folder="./forest"):
        """
        Saves a tree to a JSON file
            - save_name  : save file name (w/o '.json')
            - save_folder: folder where to save JSON trees
        """
        json_tree = json.loads(self.to_json(with_data=True))

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
            if img.endswith('.jpg'):
                test_image = load_test_image(folder=pos_image_folder, fileid=img)
                flat_output = flat_model.predict(test_image)
                # we take only the positive prediction score
                fc3_output = fc3_model.predict(test_image)[0][0]

                y_dict.update({img[:-4]: fc3_output})

                g = self.compute_g(flat_output)
                x = tf.reshape(flat_output, shape=(7, 7, 512))
                # inner product between g and x
                b = tf.subtract(fc3_output, tf.reduce_sum(
                    tf.math.multiply(g, x), axis=None))

                s = tf.math.reduce_mean(x, axis=[0, 1])
                s_list.append(s)
                self.create_node(tag=img[:-4], identifier=img[:-4],
                                parent='root', g=g, alpha=tf.ones(shape=512), b=b, x=x)
                i += 1
                print(">> created", i, "nodes") if i % 10 == 0 else \

                # TEST IF g and b ARE ACCURATE ENOUGH - IS WORKING! #
                # print("\nORIGINAL y -- CALULATED y")
                # print(fc3_output, " = ", tf.add(tf.reduce_sum(tf.math.multiply(g, x), axis=None), b).numpy())

        # sum on s to obtain s(1,1,512) / # images in the positive set
        self.s = tf.reduce_sum(s_list, axis=0)/len(fnmatch.filter(os.listdir(pos_image_folder), '*.jpg'))
        
        print("[TIME] -- init leaves took        ", dt.now()-start)
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
        start = dt.now()

        for node in self.leaves():
            # computation of x and g
            node.g = tf.multiply(tf.math.scalar_mul(1/L, self.s), vectorify_on_depth(node.g))
            node.x = tf.divide(vectorify_on_depth(node.x), self.s)
            # normalization of g and b
            norm_g = tf.norm(node.g, ord=2)
            node.b = tf.divide(node.b, norm_g)
            node.g = tf.divide(node.g, norm_g)
            # computation of w
            node.w = node.g
            # computation of gamma using normalized y_i
            gamma += y_dict[node.tag]/norm_g

        cardinality = len(self.leaves())
        self.gamma = cardinality/gamma              # gamma viene usata solo per calcolare E
        print("[TIME] -- vectorifing leaves took ", dt.now()-start)

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

    def compute_theta0(self):
        theta = 0
        start = dt.now()
        for node in self.leaves():
            node.exph_val = node.exph(self.gamma)
            theta += node.exph_val
        self.theta = theta
        print("[TIME] -- computing theta took    ", dt.now()-start)

    def compute_E0(self):
        E = 0
        start = dt.now()
        for node in self.leaves():
            E += log(node.exph_val) 
        self.E = E - len(self.leaves())*self-theta
        print("[TIME] -- computing theta took    ", dt.now()-start)
