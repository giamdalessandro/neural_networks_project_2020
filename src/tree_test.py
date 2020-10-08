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
fc_model = tf.keras.Sequential([
    m_trained.get_layer("fc1"),
    m_trained.get_layer("fc2"),
    m_trained.get_layer("fc3"),
    #m_trained.get_layer("activation")
])

N = 5
decision_paths = []
tested = 0
start = dt.now()
for img in os.listdir(POS_IMAGE_SET_TEST):
    if img.endswith('.jpg') and img[0] == '2':
        print(">> Analyzing image", img, "test", str(tested+1))
        test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
        flat_output = flat_model.predict(test_image)
        y = m_trained.predict(test_image)[0][0]

        decision_paths.append(twA.def_note(flat_output, fc_model, y))
        '''
        for i in range(len(decision_paths[-1])):
            g_outo = decision_paths[-1][str(i+1)]['g_outo']
            tab = "  "*(i+1)
            print(("  "*i)+"[LEVEL] --", str(i+1))
            print(tab, ' ├──', "Contribution of head parts  --", g_outo[0][0].numpy())
            print(tab, ' ├──', "Contribution of torso parts --", g_outo[1][0].numpy())
            print(tab, ' ├──', "Contribution of leg parts   --", g_outo[2][0].numpy())
            print(tab, ' └──', "Contribution of tail parts  --", g_outo[3][0].numpy())
            #print(tab, ' ─M1', decision_paths[-1][str(i+1)]['m1'])
            #print(tab, ' ─M2', decision_paths[-1][str(i+1)]['m2'])
            #print(tab, ' ─M3', decision_paths[-1][str(i+1)]['m3'])
            break
        #visualize_objpart_RF(m_trained, test_image, twA.A,os.path.join(POS_IMAGE_SET_TEST, img))
        '''
        tested += 1
        if tested == N:
            break
print(dt.now()-start)

# METRIC 1 #
M1_L2 = np.zeros((N, 4))
M1_L5 = np.zeros((N, 4))
M1_L9 = np.zeros((N, 4))
ys = 0

M2_L2 = np.zeros((512))
M2_L5 = np.zeros((512))
M2_L9 = np.zeros((512))

M3_L2 = 0
M3_L5 = 0
M3_L9 = 0

miny = 1
maxy = 0
m3 = 0
for i in range(len(decision_paths)):
    # M1 #
    M1_L2[i] = tf.reshape(decision_paths[i]['1']['m1'], shape=(4,))
    M1_L5[i] = tf.reshape(decision_paths[i]['4']['m1'], shape=(4,))
    if '8' in decision_paths[i]:
        M1_L9[i] = tf.reshape(decision_paths[i]['8']['m1'], shape=(4,))
    ys += decision_paths[i]['1']['pred']

    # M2 #
    M2_L2 = tf.add(M2_L2, decision_paths[i]['1']['m2'])
    M2_L5 = tf.add(M2_L5, decision_paths[i]['4']['m2'])
    if '8' in decision_paths[i]:
        M2_L9 = tf.add(M2_L9, decision_paths[i]['8']['m2'])

    # M3 #
    print("h(x)", decision_paths[i]['1']['m3'])
    M3_L2 += abs(decision_paths[i]['1']['m3'] - decision_paths[i]['1']['pred'])
    m3 += decision_paths[i]['1']['m3']
    M3_L5 += abs(decision_paths[i]['4']['m3'] - decision_paths[i]['4']['pred'])
    if '8' in decision_paths[i]:
        M3_L9 += abs(decision_paths[i]['8']['m3'] - decision_paths[i]['8']['pred'])
    
    if decision_paths[i]['1']['pred'] > maxy:
        maxy = decision_paths[i]['1']['pred']
    if decision_paths[i]['1']['pred'] < miny:
        miny = decision_paths[i]['1']['pred']

#ys = ys/len(decision_paths)                                    # tralalà
M1_L2 = tf.divide(tf.reduce_mean(tf.convert_to_tensor(M1_L2, dtype=tf.float32), axis=0), ys)
#M1_L5 = tf.divide(tf.reduce_mean(M1_L5, axis=0), ys)
#M1_L9 = tf.divide(tf.reduce_mean(M1_L9, axis=0), ys)

print("M1_L2", M1_L2.numpy())
#print("M1_L5", M1_L5.numpy())
#print("M1_L9", M1_L9.numpy())
print("----------------------------------------")
print("----------------------------------------")
print("----------------------------------------")

M2_L2 = tf.reduce_mean(M2_L2)
M2_L5 = tf.reduce_mean(M2_L5)
M2_L9 = tf.reduce_mean(M2_L9)

print("M2_L2", M2_L2.numpy())
print("M2_L5", M2_L5.numpy())
print("M2_L9", M2_L9.numpy())

print("----------------------------------------")
print("----------------------------------------")
print("----------------------------------------")

M3_L2 = M3_L2/(maxy - miny)
#M3_L5 = M3_L5/(maxy - miny)
#M3_L9 = M3_L9/(maxy - miny)
print("M3_L2 - ERROR", M3_L2.numpy())
print("M3_L2 ", m3)
#print("M3_L5", M3_L5.numpy())
#print("M3_L9", M3_L9.numpy())


'''
TODO:
    - test metrics albero fake
    - computazione metrics su albero vero
    - printare fiorellini albero vero
    - scrivere slide e report
    - scrivere README
'''
