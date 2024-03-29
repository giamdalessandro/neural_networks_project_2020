import tensorflow as tf
from utils.A_utils import *
from classes.tree_utils import *
from classes.maskLayer import MaskLayer
from classes.interpretableNode import InterpretableNode
from classes.interpretableTree import InterpretableTree
from sklearn.metrics import accuracy_score, jaccard_score


def gimme_g_gimme_x(test_image, flat_model, fc_model, s):
    flat_x = flat_model.predict(test_image)
    x = tf.reshape(flat_x, shape=(7, 7, 512))
    x = tf.divide(vectorify_on_depth(x), s)
    x = tf.reshape(x, shape=(512, 1))

    g = compute_g(fc_model, flat_x)
    g = tf.multiply(tf.math.scalar_mul(1/L, s), vectorify_on_depth(g))
    g = tf.divide(g, tf.norm(g, ord=2))
    return x, g


start = dt.now()
METRICS = 4

TREE100 = "./forest/test_tree_100_imgs_with_A.json"
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")

# partially resolves CUBLAS errors
gpus = tf.config.experimental.list_physical_devices('GPU')  
tf.config.experimental.set_memory_growth(gpus[0], True)

# load saved models and tree100 #
#with tf.device("/CPU:0"):
cnn = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer": MaskLayer()})
flat_model = Model(inputs=cnn.input, outputs=cnn.get_layer("flatten").output)
fc_model = tf.keras.Sequential([
    cnn.get_layer("fc1"),
    cnn.get_layer("fc2"),
    cnn.get_layer("fc3")
])


tree100 = from_json(InterpretableTree(), TREE100)
tree100.info()


# metrica 1 #
if METRICS == 1:
    fc_model_activated = tf.keras.Sequential([
        cnn.get_layer("fc1"),
        cnn.get_layer("fc2"),
        cnn.get_layer("fc3"),
        cnn.get_layer("activation")
    ])
    BREAK = 100
    POS_IMAGE_SET_TEST = "./dataset/train_val/bird"
    
    y_list = []
    g_strano_1 = []
    g_strano_2 = []
    g_strano_3 = []
    g_strano_4 = []
    g_strano_5 = []
    g_strano_6 = []
    g_strano_leaves = []
    
    i = 0
    for img in os.listdir(POS_IMAGE_SET_TEST):
        if img.endswith('.jpg') and img[0] == '2':
            test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
            
            y = cnn.predict(test_image)[0][0]
            y_list.append(y)
            x, g = gimme_g_gimme_x(test_image, flat_model, fc_model, tree100.s)
            
            flat_x = flat_model.predict(test_image)
            flat_x = tf.reshape(flat_x, shape=(7, 7, 512))
            q = []
            for p in range(4):
                x_p = tf.multiply(flat_x, tf.reshape(tree100.A, shape=(4, 512)).numpy()[p])
                y_p = fc_model_activated.predict(tf.reshape(x_p, shape=(1, 25088)))[0][0]
                q.append(y-y_p)
            q = tf.reshape(tf.convert_to_tensor(q), shape=(4, 1))
            

            g_strano_1.append(     tf.abs(tf.subtract(tree100.compute_g_strano(x, g, level=1),  q)))
            g_strano_2.append(     tf.abs(tf.subtract(tree100.compute_g_strano(x, g, level=2),  q)))
            g_strano_3.append(     tf.abs(tf.subtract(tree100.compute_g_strano(x, g, level=3),  q)))
            g_strano_4.append(     tf.abs(tf.subtract(tree100.compute_g_strano(x, g, level=4),  q)))
            g_strano_5.append(     tf.abs(tf.subtract(tree100.compute_g_strano(x, g, level=5),  q)))
            g_strano_6.append(     tf.abs(tf.subtract(tree100.compute_g_strano(x, g, level=6),  q)))
            g_strano_leaves.append(tf.abs(tf.subtract(tree100.compute_g_strano(x, g, level=-1), q)))
            
            i += 1
        if i == BREAK:
            break
    
    ymean = tf.reduce_mean(y_list)
    print("                                        | HEAD --- |TORSO --- |LEG ---   |TAIL ---   |AVG")
    print("objpart contribution error layer 1     ", tf.reshape(tf.divide(tf.reduce_mean(g_strano_1,      axis=0), ymean), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.divide(tf.reduce_mean(g_strano_1,      axis=0), ymean), shape=(4))).numpy())
    print("objpart contribution error layer 2     ", tf.reshape(tf.divide(tf.reduce_mean(g_strano_2,      axis=0), ymean), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.divide(tf.reduce_mean(g_strano_2,      axis=0), ymean), shape=(4))).numpy())
    print("objpart contribution error layer 3     ", tf.reshape(tf.divide(tf.reduce_mean(g_strano_3,      axis=0), ymean), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.divide(tf.reduce_mean(g_strano_3,      axis=0), ymean), shape=(4))).numpy())
    print("objpart contribution error layer 4     ", tf.reshape(tf.divide(tf.reduce_mean(g_strano_4,      axis=0), ymean), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.divide(tf.reduce_mean(g_strano_4,      axis=0), ymean), shape=(4))).numpy())
    print("objpart contribution error layer 5     ", tf.reshape(tf.divide(tf.reduce_mean(g_strano_5,      axis=0), ymean), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.divide(tf.reduce_mean(g_strano_5,      axis=0), ymean), shape=(4))).numpy())
    print("objpart contribution error layer 6     ", tf.reshape(tf.divide(tf.reduce_mean(g_strano_6,      axis=0), ymean), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.divide(tf.reduce_mean(g_strano_6,      axis=0), ymean), shape=(4))).numpy())
    print("objpart contribution error layer leaves", tf.reshape(tf.divide(tf.reduce_mean(g_strano_leaves, axis=0), ymean), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.divide(tf.reduce_mean(g_strano_leaves, axis=0), ymean), shape=(4))).numpy())
    
    print("                                        | HEAD --- |TORSO --- |LEG ---   |TAIL ---   |AVG")
    print("Objpart contribution layer 1:          ", tf.reshape(tf.reduce_mean(g_strano_1     , axis=0), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.reduce_mean(g_strano_1     , axis=0), shape=(4))).numpy())
    print("Objpart contribution layer 2:          ", tf.reshape(tf.reduce_mean(g_strano_2     , axis=0), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.reduce_mean(g_strano_2     , axis=0), shape=(4))).numpy())
    print("Objpart contribution layer 3:          ", tf.reshape(tf.reduce_mean(g_strano_3     , axis=0), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.reduce_mean(g_strano_3     , axis=0), shape=(4))).numpy())
    print("Objpart contribution layer 4:          ", tf.reshape(tf.reduce_mean(g_strano_4     , axis=0), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.reduce_mean(g_strano_4     , axis=0), shape=(4))).numpy())
    print("Objpart contribution layer 5:          ", tf.reshape(tf.reduce_mean(g_strano_5     , axis=0), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.reduce_mean(g_strano_5     , axis=0), shape=(4))).numpy())
    print("Objpart contribution layer 6:          ", tf.reshape(tf.reduce_mean(g_strano_6     , axis=0), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.reduce_mean(g_strano_6     , axis=0), shape=(4))).numpy())
    print("Objpart contribution layer leaves:     ", tf.reshape(tf.reduce_mean(g_strano_leaves, axis=0), shape=(4)).numpy(), tf.reduce_mean(tf.reshape(tf.reduce_mean(g_strano_leaves, axis=0), shape=(4))).numpy())


# metrica 2 #
if METRICS == 2: 
    BREAK = 100
    POS_IMAGE_SET_TEST = "./dataset/train_val/bird"
    
    hatrho_1 = []
    hatrho_2 = []
    hatrho_3 = []
    hatrho_4 = []
    hatrho_5 = []
    hatrho_6 = []
    hatrho_leaves = []
    
    i = 0
    for img in os.listdir(POS_IMAGE_SET_TEST):
        if img.endswith('.jpg') and img[0] == '2':
            test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
            x, g = gimme_g_gimme_x(test_image, flat_model, fc_model, tree100.s)
            
            t = tf.multiply(g, x)
            print("I am legend, ",i)
            
            hatrho_1.append(tf.divide(tf.reduce_sum(tf.minimum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=1))),
                                      tf.reduce_sum(tf.maximum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=1)))))

            hatrho_2.append(tf.divide(tf.reduce_sum(tf.minimum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=2))),
                                      tf.reduce_sum(tf.maximum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=2)))))
            
            hatrho_3.append(tf.divide(tf.reduce_sum(tf.minimum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=3))),
                                      tf.reduce_sum(tf.maximum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=3)))))
            
            hatrho_4.append(tf.divide(tf.reduce_sum(tf.minimum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=4))),
                                      tf.reduce_sum(tf.maximum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=4)))))
            
            hatrho_5.append(tf.divide(tf.reduce_sum(tf.minimum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=5))),
                                      tf.reduce_sum(tf.maximum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=5)))))
            
            hatrho_6.append(tf.divide(tf.reduce_sum(tf.minimum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=6))),
                                      tf.reduce_sum(tf.maximum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=6)))))
            
            hatrho_leaves.append(tf.divide(tf.reduce_sum(tf.minimum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=-1))),
                                           tf.reduce_sum(tf.maximum(tf.abs(t), tree100.compute_hatrho(x, g, t, level=-1)))))

            i += 1
        if i == BREAK:
            break

    print("jaccard score TREE layer 1     ", tf.reduce_mean(hatrho_1).numpy())
    print("jaccard score TREE layer 2     ", tf.reduce_mean(hatrho_2).numpy())
    print("jaccard score TREE layer 3     ", tf.reduce_mean(hatrho_3).numpy())
    print("jaccard score TREE layer 4     ", tf.reduce_mean(hatrho_4).numpy())
    print("jaccard score TREE layer 5     ", tf.reduce_mean(hatrho_5).numpy())
    print("jaccard score TREE layer 6     ", tf.reduce_mean(hatrho_6).numpy())
    print("jaccard score TREE layer leaves", tf.reduce_mean(hatrho_leaves).numpy())

# metrica 3 #
if METRICS == 3:
    POS_IMAGE_SET_TEST = "./dataset/train_val/bird"
    NEG_IMAGE_SET_TEST = "./dataset/train_val/not_bird"
    BREAK = 100
    MEAN = 0.5

    y_true = []
    y_cnn  = []
    y_tree100_1 = []
    y_tree100_2 = []
    y_tree100_3 = []
    y_tree100_4 = []
    y_tree100_5 = []
    y_tree100_6 = []
    y_tree100_leaves = []
    ymax = 0
    ymin = 1

    err1 = []
    err2 = []
    err3 = []
    err4 = []
    err5 = []
    err6 = []
    errleaves = []

    print(">> Testing on ", BREAK, " positive images")
    i = 0
    for img in os.listdir(POS_IMAGE_SET_TEST):
        if img.endswith('.jpg') and img[0] == '2':
            test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
            x, g = gimme_g_gimme_x(test_image, flat_model, fc_model, tree100.s)

            y_true.append(1)
            y = cnn.predict(test_image)[0][0]
            if y > ymax:
                ymax = y
            if y < ymin:
                ymin = y
            
            y_cnn.append(1 if y > 0.5 else 0)

            y1      = tree100.predict(g, x, level=1)
            y2      = tree100.predict(g, x, level=2)
            y3      = tree100.predict(g, x, level=3)
            y4      = tree100.predict(g, x, level=4)
            y5      = tree100.predict(g, x, level=5)
            y6      = tree100.predict(g, x, level=6)
            yleaves = tree100.predict(g, x, level=-1)

            y_tree100_1.append(     1 if y1      > MEAN else 0)
            y_tree100_2.append(     1 if y2      > MEAN else 0)
            y_tree100_3.append(     1 if y3      > MEAN else 0)
            y_tree100_4.append(     1 if y4      > MEAN else 0)
            y_tree100_5.append(     1 if y5      > MEAN else 0)
            y_tree100_6.append(     1 if y6      > MEAN else 0)
            y_tree100_leaves.append(1 if yleaves > MEAN else 0)
            
            err1.append(     abs(y1      - y))
            err2.append(     abs(y2      - y))
            err3.append(     abs(y3      - y))
            err4.append(     abs(y4      - y))
            err5.append(     abs(y5      - y))
            err6.append(     abs(y6      - y))
            errleaves.append(abs(yleaves - y))

            i += 1
            if i == BREAK:
                break
    print(">> Testing on ", BREAK, " negative images")
    i = 0
    for img in os.listdir(NEG_IMAGE_SET_TEST):
        if img.endswith('.jpg') and img[0] == '2'::
            test_image = load_test_image(folder=NEG_IMAGE_SET_TEST, fileid=img)
            x, g = gimme_g_gimme_x(test_image, flat_model, fc_model, tree100.s)

            y_true.append(0)
            y = cnn.predict(test_image)[0][0]
            if y > ymax:
                ymax = y
            if y < ymin:
                ymin = y
            
            y_cnn.append(1 if y > 0.5 else 0)
            y1      = tree100.predict(g, x, level=1)
            y2      = tree100.predict(g, x, level=2)
            y3      = tree100.predict(g, x, level=3)
            y4      = tree100.predict(g, x, level=4)
            y5      = tree100.predict(g, x, level=5)
            y6      = tree100.predict(g, x, level=6)
            yleaves = tree100.predict(g, x, level=-1)

            y_tree100_1.append(     1 if y1      > MEAN else 0)
            y_tree100_2.append(     1 if y2      > MEAN else 0)
            y_tree100_3.append(     1 if y3      > MEAN else 0)
            y_tree100_4.append(     1 if y4      > MEAN else 0)
            y_tree100_5.append(     1 if y5      > MEAN else 0)
            y_tree100_6.append(     1 if y6      > MEAN else 0)
            y_tree100_leaves.append(1 if yleaves > MEAN else 0)
            
            err1.append(     abs(y1      - y))
            err2.append(     abs(y2      - y))
            err3.append(     abs(y3      - y))
            err4.append(     abs(y4      - y))
            err5.append(     abs(y5      - y))
            err6.append(     abs(y6      - y))
            errleaves.append(abs(yleaves - y))

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
    print("accuracy_score TREE leaves  ", (accuracy_score(y_true, y_tree100_leaves)*100))
    #ymax = 1
    #ymin = 0
    print("prediction error 2nd  layer", tf.reduce_mean(tf.divide(err1     , ymax-ymin)).numpy())
    print("prediction error 3rd  layer", tf.reduce_mean(tf.divide(err2     , ymax-ymin)).numpy())
    print("prediction error 4th  layer", tf.reduce_mean(tf.divide(err3     , ymax-ymin)).numpy())
    print("prediction error 5th  layer", tf.reduce_mean(tf.divide(err4     , ymax-ymin)).numpy())
    print("prediction error 6th  layer", tf.reduce_mean(tf.divide(err5     , ymax-ymin)).numpy())
    print("prediction error 7th  layer", tf.reduce_mean(tf.divide(err6     , ymax-ymin)).numpy())
    print("prediction error leaves    ", tf.reduce_mean(tf.divide(errleaves, ymax-ymin)).numpy())

if METRICS == 4:
    POS_IMAGE_SET_TEST = "./dataset/train_val/bird"
    images = ["2008_003211.jpg", "2008_004730.jpg", "2008_004893.jpg",
              "2008_005803.jpg", "2008_005895.jpg", "2008_006164.jpg", "2008_006281.jpg", "2008_006376.jpg", "2008_006626.jpg", "2008_008284.jpg", "2008_008423.jpg", "2008_008404.jpg"]
    #[ "2008_000512.jpg", "2008_001679.jpg", "2008_001810.jpg", "2008_002389.jpg", "2008_003023.jpg"]
    for img in images:
        test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
        visualize_objpart_RF(cnn, test_image, tree100.A, os.path.join(POS_IMAGE_SET_TEST, img), img)

print("TIME --", dt.now()-start)
print("þøþł")
