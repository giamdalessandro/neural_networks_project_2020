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

# gpus = tf.config.experimental.list_physical_devices('GPU')          #    partially resolves CUBLAS errors
# tf.config.experimental.set_memory_growth(gpus[0], True)

with tf.device("/CPU:0"):
    m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})

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
else:
    with tf.device("/CPU:0"):
        loaded = from_json(InterpretableTree(), "./forest/test_tree_"+str(STOP)+"_imgs.json")
        loaded.A = compute_A(POS_IMAGE_SET_TEST, stop=100)
        saved = loaded.save2json(save_name="test_tree_"+str(STOP)+"_imgs_with_A")
        print("A done.")
        print("Evvai.")

with tf.device("/CPU:0"):
    twA = from_json(InterpretableTree(), "./forest/test_tree_"+str(STOP)+"_imgs_with_A.json")
    flat_model = Model(inputs=m_trained.input, outputs=m_trained.get_layer("flatten").output)
    
    test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid="2010_005603.jpg")
    flat_output = flat_model.predict(test_image)
    y = m_trained.predict(test_image)[0][0]
    
    rho = twA.def_note(flat_output, m_trained)["1"]["rho"]
    g_outo = twA.def_note(flat_output, m_trained)["1"]["g_outo"]
    g_outo = tf.multiply(100/tf.reduce_sum(g_outo), g_outo)
    print(rho)
    print("Contribution of head parts  --", g_outo[0][0].numpy())
    print("Contribution of torso parts --", g_outo[1][0].numpy())
    print("Contribution of leg parts   --", g_outo[2][0].numpy())
    print("Contribution of tail parts  --", g_outo[3][0].numpy())

visualize_objpart_RF(m_trained, test_image, twA.A,os.path.join(POS_IMAGE_SET_TEST, "2010_005603.jpg"))
