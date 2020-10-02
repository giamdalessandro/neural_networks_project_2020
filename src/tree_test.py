import tensorflow as tf
from classes.tree_utils import *
from classes.interpretableNode import InterpretableNode
from classes.interpretableTree import InterpretableTree

from classes.maskLayer import MaskLayer

MODELS  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")
TEST = False

# partially resolves CUBLAS errors
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#with tf.device("/CPU:0"):
m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})

POS_IMAGE_SET_TEST = "./dataset/train_val/test/bird/"
start = dt.now()

tree, y_dict, x_dict = sow(m_trained, POS_IMAGE_SET_TEST)
tree.info()
#tree.show()

new_tree = grow(tree, y_dict, x_dict)
new_tree.info()
new_tree.show()

saved = tree.save2json(save_name="test_tree_"+str(STOP)+"_imgs")
# loaded = from_json(InterpretableTree(), "./forest/test_tree.json")

print("[TIME] -- test on ", STOP, " images took ", dt.now()-start)

'''
TODO
    - scrivere "find_gab"       - ro
    - scrivere "e_func"         - balthier
    - calcolare matrice A       - ???
'''

