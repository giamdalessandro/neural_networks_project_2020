import tensorflow as tf
from utils.A_utils import *
from classes.tree_utils import *
from classes.interpretableNode import InterpretableNode
from classes.interpretableTree import InterpretableTree
from classes.maskLayer import MaskLayer

TREE = "./forest/test_tree_100_imgs_with_A.json"
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")

# partially resolves CUBLAS errors
gpus = tf.config.experimental.list_physical_devices('GPU')  
tf.config.experimental.set_memory_growth(gpus[0], True)

# load saved models and tree #
m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer": MaskLayer()})
tree = from_json(InterpretableTree(), TREE)
tree.info()


# metrica 3 - dobbiamo passare una lista di prediction e di ground truth
POS_IMAGE_SET_TEST = "./dataset/train_val/evaluation_test/bird"
NEG_IMAGE_SET_TEST = "./dataset/train_val/evaluation_test/not_bird"

for img in dir
    cnn.predict
    tree.predict

accuration score
accuration error














print("þøþł")
