from sklearn.metrics import accuracy_score
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
cnn = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer": MaskLayer()})
tree = from_json(InterpretableTree(), TREE)
tree.info()


# metrica 3 - dobbiamo passare una lista di prediction e di ground truth
POS_IMAGE_SET_TEST = "./dataset/evaluation_test/bird"
NEG_IMAGE_SET_TEST = "./dataset/evaluation_test/not_bird"

y_true = []
y_cnn  = []
y_tree = []

for img in os.listdir(POS_IMAGE_SET_TEST):
    test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
    y_true.append(1)
    y_cnn.append(1 if cnn.predict(test_image)[0][0] > 0.5 else 0)
    y_tree.append(1 if tree.predict(test_image, cnn) > 0.5 else 0)

for img in os.listdir(NEG_IMAGE_SET_TEST):
    test_image = load_test_image(folder=NEG_IMAGE_SET_TEST, fileid=img)
    y_true.append(0)
    y_cnn.append(1 if cnn.predict(test_image)[0][0] > 0.5 else 0)
    y_tree.append(1 if tree.predict(test_image, cnn) > 0.5 else 0)

print("accuracy_score CNN ", (accuracy_score(y_true, y_cnn)*100))
print("accuracy_score TREE", (accuracy_score(y_true, y_tree)*100))



'''
def contrib_fitness(y_true, y_pred):

    return jaccard_similarity_score(y_true, y_pred)

'''











print("þøþł")
