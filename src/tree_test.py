import tensorflow as tf
from utils.A_utils import *
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

##with tf.device("/CPU:0"):
#m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})
#if STOP <= 20:
#    POS_IMAGE_SET_TEST = "./dataset/train_val/test/bird/"
#else:
#    POS_IMAGE_SET_TEST = "./dataset/train_val/bird/"
#
#POS_IMAGE_SET_TEST = "./dataset/train_val/bird/"
#start = dt.now()
#tree, y_dict, x_dict = sow(m_trained, POS_IMAGE_SET_TEST)
#tree.info()
##tree.show()
#
#new_tree = grow(tree, y_dict, x_dict)
#new_tree.info()
#new_tree.show()
#
#saved = tree.save2json(save_name="test_tree_"+str(STOP)+"_imgs")
#loaded = from_json(InterpretableTree(), "./forest/test_tree_20._imgs.json")

#print("[TIME] -- test on ", STOP, " images took ", dt.now()-start)


# CODE FOR COMPUTING AND SAVING A #
A = compute_A(stop=10)   # stop = 0 means it will do all images
loaded = from_json(InterpretableTree(), "./forest/test_tree_100_imgs.json")
loaded.A = A
saved = loaded.save2json(save_name="./forest/test_tree_100_imgs_with_A")
print("A done.")


print("Evvai.")


'''
TODO
    - calcolare matrice A       - ???
'''

