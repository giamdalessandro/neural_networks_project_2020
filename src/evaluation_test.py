import tensorflow as tf
from utils.A_utils import *
from classes.tree_utils import *
from classes.maskLayer import MaskLayer
from classes.interpretableNode import InterpretableNode
from classes.interpretableTree import InterpretableTree
from sklearn.metrics import accuracy_score, jaccard_similarity_score


TREE100 = "./forest/test_tree_100_imgs_with_A.json"
TREE007 = "./forest/test_tree_7_imgs_with_A.json"
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")

# partially resolves CUBLAS errors
gpus = tf.config.experimental.list_physical_devices('GPU')  
tf.config.experimental.set_memory_growth(gpus[0], True)

# load saved models and tree100 #
cnn = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer": MaskLayer()})
flat_model = Model(inputs=cnn.input, outputs=cnn.get_layer("flatten").output)
fc_model = tf.keras.Sequential([
    cnn.get_layer("fc1"),
    cnn.get_layer("fc2"),
    cnn.get_layer("fc3")
])


tree100 = from_json(InterpretableTree(), TREE100)
tree100.info()



# metrica 3 #
POS_IMAGE_SET_TEST = "./dataset/train_val/bird"
NEG_IMAGE_SET_TEST = "./dataset/train_val/not_bird"
BREAK = 100000

y_true = []
y_cnn  = []
y_tree100_1 = []
y_tree100_2 = []
y_tree100_3 = []
y_tree100_4 = []
y_tree100_5 = []
y_tree100_6 = []
y_tree100_8 = []

start = dt.now()
print(">> Testing on ", BREAK, " positive images")
i = 0
for img in os.listdir(POS_IMAGE_SET_TEST):
    if img.endswith('.jpg'):
        test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
        flat_output = flat_model.predict(test_image)

        y_true.append(1)
        y_cnn.append(1 if cnn.predict(test_image)[0][0] > 0.5 else 0)
        y_tree100_1.append(1 if tree100.predict(test_image, fc_model, flat_output, level=1) > 0.5 else 0)
        y_tree100_2.append(1 if tree100.predict(test_image, fc_model, flat_output, level=2) > 0.5 else 0)
        y_tree100_3.append(1 if tree100.predict(test_image, fc_model, flat_output, level=3) > 0.5 else 0)
        y_tree100_4.append(1 if tree100.predict(test_image, fc_model, flat_output, level=4) > 0.5 else 0)
        y_tree100_5.append(1 if tree100.predict(test_image, fc_model, flat_output, level=5) > 0.5 else 0)
        y_tree100_6.append(1 if tree100.predict(test_image, fc_model, flat_output, level=6) > 0.5 else 0)
        y_tree100_8.append(1 if tree100.predict(test_image, fc_model, flat_output, level=-1) > 0.5 else 0)
        i += 1
        if i == BREAK:
            break

print(">> Testing on ", BREAK, " negative images")
i = 0
for img in os.listdir(NEG_IMAGE_SET_TEST):
    if img.endswith('.jpg'):
        test_image = load_test_image(folder=NEG_IMAGE_SET_TEST, fileid=img)
        y_true.append(0)
        y_cnn.append(1 if cnn.predict(test_image)[0][0] > 0.5 else 0)
        y_tree100_1.append(1 if tree100.predict(test_image, fc_model, flat_output, level=1) > 0.5 else 0)
        y_tree100_2.append(1 if tree100.predict(test_image, fc_model, flat_output, level=2) > 0.5 else 0)
        y_tree100_3.append(1 if tree100.predict(test_image, fc_model, flat_output, level=3) > 0.5 else 0)
        y_tree100_4.append(1 if tree100.predict(test_image, fc_model, flat_output, level=4) > 0.5 else 0)
        y_tree100_5.append(1 if tree100.predict(test_image, fc_model, flat_output, level=5) > 0.5 else 0)
        y_tree100_6.append(1 if tree100.predict(test_image, fc_model, flat_output, level=6) > 0.5 else 0)
        y_tree100_8.append(1 if tree100.predict(test_image, fc_model, flat_output, level=-1) > 0.5 else 0)
        i += 1
        if i == BREAK:
            break


print("accuracy_score CNN          ", (accuracy_score(y_true, y_cnn)*100))
print("accuracy_score TREE layer 1 ", (accuracy_score(y_true, y_tree100_1)*100))
print("accuracy_score TREE layer 2 ", (accuracy_score(y_true, y_tree100_2)*100))
print("accuracy_score TREE layer 3 ", (accuracy_score(y_true, y_tree100_3)*100))
print("accuracy_score TREE layer 4 ", (accuracy_score(y_true, y_tree100_4)*100))
print("accuracy_score TREE layer 5 ", (accuracy_score(y_true, y_tree100_5)*100))
print("accuracy_score TREE layer 6 ", (accuracy_score(y_true, y_tree100_6)*100))
print("accuracy_score TREE leaves  ", (accuracy_score(y_true, y_tree100_8)*100))

print("TIME --", dt.now()-start)












print("þøþł")
