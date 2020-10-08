import tensorflow as tf
from utils.A_utils import *
from classes.tree_utils import *
from classes.interpretableNode import InterpretableNode
from classes.interpretableTree import InterpretableTree
from classes.maskLayer import MaskLayer

POS_IMAGE_SET_TEST = "./dataset/train_val/bird"
MODELS  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")

TEST = False
COMPUTE_A = False

gpus = tf.config.experimental.list_physical_devices('GPU')          #    partially resolves CUBLAS errors
tf.config.experimental.set_memory_growth(gpus[0], True)

m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})

# CODE FOR CREATING THE TREE #
if TEST:
    start = dt.now()
    tree, y_dict, x_dict = sow(m_trained, POS_IMAGE_SET_TEST)
    tree.info()

    new_tree = grow(tree, y_dict, x_dict)
    new_tree.info()
    new_tree.show()
    saved = new_tree.save2json(save_name="test_tree_"+str(STOP)+"_imgs")
    print("[TIME] -- test on ", STOP, " images took ", dt.now()-start)

# CODE FOR COMPUTING AND SAVING A #
if COMPUTE_A:
    loaded = from_json(InterpretableTree(), "./forest/test_tree_"+str(STOP)+"_imgs.json")
    loaded.A = compute_A(POS_IMAGE_SET_TEST, stop=100)
    saved = loaded.save2json(save_name="test_tree_"+str(STOP)+"_imgs_with_A")
    print("A done.")
    print("Evvai.")


# COMPUTES RHO AND G OUTO
twA = from_json(InterpretableTree(), "./forest/test_tree_"+str(STOP)+"_imgs_with_A.json")
twA.info()

flat_model = Model(inputs=m_trained.input, outputs=m_trained.get_layer("flatten").output)


decision_paths = []
for img in os.listdir(POS_IMAGE_SET_TEST):
    if img.endswith('.jpg') and img[0] == '2':
        print(">> Analyzing image", img)
        test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
        flat_output = flat_model.predict(test_image)
        y = m_trained.predict(test_image)[0][0]

        decision_paths.append(twA.def_note(flat_output, m_trained))

        for i in range(len(decision_paths[-1])):
            g_outo = decision_paths[-1][str(i+1)]['g_outo']
            tab = "  "*(i+1)
            print(("  "*i)+"[LEVEL] --", str(i+1))
            print(tab, ' ├──', "Contribution of head parts  --", g_outo[0][0].numpy())
            print(tab, ' ├──', "Contribution of torso parts --", g_outo[1][0].numpy())
            print(tab, ' ├──', "Contribution of leg parts   --", g_outo[2][0].numpy())
            print(tab, ' └──', "Contribution of tail parts  --", g_outo[3][0].numpy())
            print(tab, '────', decision_paths[-1][str(i+1)]['m1'])
            print(tab, '────', decision_paths[-1][str(i+1)]['m2'])
            print(tab, '────', decision_paths[-1][str(i+1)]['m3'])
        visualize_objpart_RF(m_trained, test_image, twA.A,os.path.join(POS_IMAGE_SET_TEST, img))
        break
        







'''
TODO:
    - test metrics albero fake
    - computazione metrics su albero vero
    - printare fiorellini albero vero
    - scrivere slide e report
    - scrivere README
'''
