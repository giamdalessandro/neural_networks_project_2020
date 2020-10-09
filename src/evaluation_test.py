import tensorflow as tf
from utils.A_utils import *
from classes.tree_utils import *
from classes.interpretableNode import InterpretableNode
from classes.interpretableTree import InterpretableTree
from classes.maskLayer import MaskLayer

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")

POS_IMAGE_SET_TEST = "./dataset/train_val/bird"

# partially resolves CUBLAS errors
gpus = tf.config.experimental.list_physical_devices('GPU')  
tf.config.experimental.set_memory_growth(gpus[0], True)

m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer": MaskLayer()})

print("þøþł")