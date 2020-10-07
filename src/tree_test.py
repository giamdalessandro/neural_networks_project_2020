import tensorflow as tf
from utils.A_utils import *
from classes.tree_utils import *
from classes.interpretableNode import InterpretableNode
from classes.interpretableTree import InterpretableTree
from classes.maskLayer import MaskLayer

POS_IMAGE_SET_TEST = "./dataset/train_val/bird/test100"
MODELS  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")
TEST = False

gpus = tf.config.experimental.list_physical_devices('GPU')          #    partially resolves CUBLAS errors
tf.config.experimental.set_memory_growth(gpus[0], True)

m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})

start = dt.now()
tree, y_dict, x_dict = sow(m_trained, POS_IMAGE_SET_TEST)
tree.info()

new_tree = grow(tree, y_dict, x_dict)
new_tree.info()
new_tree.show()
saved = new_tree.save2json(save_name="test_tree_"+str(STOP)+"_imgs")
print("[TIME] -- test on ", STOP, " images took ", dt.now()-start)


# CODE FOR COMPUTING AND SAVING A #
new_tree.A = compute_A(stop=10)   # stop = 0 means it will do all images
saved = new_tree.save2json(save_name="test_tree_"+str(STOP)+"_imgs_with_A")
print("A done.")
print("Evvai.")
