import os
import cv2
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from tensorflow.keras import Model
from datetime import datetime as dt
from tensorflow.keras.preprocessing.image import img_to_array
from classes.maskLayer import MaskLayer
from classes.tree_utils import load_test_image
from utils.receptvie_field import receptive_field

MODELS  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")
STOP = 10
HEAD_PARTS   = 0
TORSO_PARTS  = 1
LEG_PARTS    = 2
TAIL_PARTS   = 3

parts = {'HEAD_PARTS':  ['head', 'beak', 'leye', 'reye'],
         'TORSO_PARTS': ['torso', 'neck', 'lwing', 'rwing'],
         'LEG_PARTS':   ['lleg', 'rleg', 'lfoot', 'rfoot'],
         'TAIL_PARTS':  ['tail']}

THRESOLD = 20   # number of max distance in pixel between two centers (annotation's and RF's) 
RF_SIZE = 100
NUM_FILTERS = 512
NUM_OBJ_PARTS = 4


ANNOTATIONS = "./dataset/raw_data/PascalVOC_2010_part/annotations_trainval/Annotations_Part/"


def read_part_annotations(img_name, anno_path=ANNOTATIONS):
    anno_file = os.path.join(anno_path, img_name[:-4] + ".mat")
    anno = loadmat(anno_file)
    categories = anno["anno"][0][0][1][0]     # matlab shit

    anno_dict = {}
    for c in categories:
        if len(c[3]) <= 0:
            return None
        cat = c[0][0]
        if cat in anno_dict:
            anno_dict[cat]["bbox"].append(c[2][0])
            for p in list(c[3][0]):
                norm = (p[0][0],p[1])
                anno_dict[cat]["parts"].append(norm)

        else:
            anno_dict[cat] = {
                "cat_id": c[1][0][0],
                "bbox": [c[2][0]],
                "parts": []
            }
            for p in list(c[3][0]):
                norm = (p[0][0],p[1])
                anno_dict[cat]["parts"].append(norm)

    return anno_dict
    

def binarify(matrix):
    for f in range(NUM_FILTERS):
        if np.sum(matrix[f]) > 0:
            l = np.argmax(matrix[f])
            matrix[f] = [0, 0, 0, 0]
            matrix[f][l] =  1
            
def find_a_center(annotation, dataset_folder):
    mask = cv2.resize(annotation[1], (224, 224), interpolation=cv2.INTER_LINEAR)
    previ = 0
    prevj = 0
    maxx  = 0
    maxy  = 0
    minx  = 224
    miny  = 224
    for i in range(224):
        for j in range(224):
            if mask[previ, j] == 0 and mask[i, j] == 1 and i < minx:
                minx = i
            if mask[previ, j] == 1 and mask[i, j] == 0 and i > maxx:
                maxx = i
            if mask[i, prevj] == 0 and mask[i, j] == 1 and j < miny:
                miny = j
            if mask[i, prevj] == 1 and mask[i, j] == 0 and j > maxy:
                maxy = j
            prevj = j
        previ = i
    x = int((maxx - minx)*0.5 + minx)
    y = int((maxy - miny)*0.5 + miny)
    # print("Annotation part", annotation[0], y, x)
    #image = cv2.resize(cv2.imread(dataset_folder+img), (224, 224), interpolation=cv2.INTER_LINEAR)
    #masked_image = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow(str(annotation[0]), masked_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return [y, x]    # leave this this way   

def updateA(A, f, obj_part):
    if obj_part is not None:
        if obj_part in parts['HEAD_PARTS']:
            A[f][HEAD_PARTS] += 1
        
        elif obj_part in parts['TORSO_PARTS']:
            A[f][TORSO_PARTS] += 1
        
        elif obj_part in parts['LEG_PARTS']:
            A[f][LEG_PARTS] += 1
        
        elif obj_part in parts['TAIL_PARTS']:
            A[f][TAIL_PARTS] += 1

        else:
            print("[ERRO] :: didn't know how to handle", obj_part)
            return
    else:
        # print("[WARN] :: no obj part matching for filter #", f)
        pass

    #print(A[f])


def compute_A(dataset_folder, stop=STOP):
    """
    Computes A binary matrix
    """""
    start = dt.now()
    m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer": MaskLayer()})
    max_pool_model = Model(inputs=m_trained.input,outputs=m_trained.get_layer("final_max_pool").output)
    A = np.zeros(shape=(512, 4))
    i = 0
    print("searchinig in ", dataset_folder)
    for img in os.listdir(dataset_folder):
        if img.endswith('.jpg') and img[0] == '2':
            print(">> Analyzing image", img)
            test_image = load_test_image(folder=dataset_folder, fileid=img)
            pool_output = max_pool_model.predict(test_image)
            rows_idx = tf.math.argmax(tf.reduce_max(pool_output[0], axis=1), output_type=tf.int32)
            cols_idx = tf.math.argmax(tf.reduce_max(pool_output[0], axis=0), output_type=tf.int32)
            annotations = read_part_annotations(img)
            if annotations is not None:
                a_centers = []
                for a in annotations['bird']['parts']:
                    a_centers.append([a[0], find_a_center(a, dataset_folder)])
                    for d in range(512):
                        max_i = rows_idx[d].numpy()
                        max_j = cols_idx[d].numpy()
                        rf_center, rf_size = receptive_field((max_i, max_j))
                        mindist = THRESOLD
                        obj_part = None
                        center = None
                        for c in range(len(a_centers)):
                            aux = abs(rf_center[0] - a_centers[c][1][0]) + abs(rf_center[1] - a_centers[c][1][1])
                            if aux < mindist:
                                mindist = aux
                                obj_part = a_centers[c][0]
                                center = a_centers[c][1]
                        updateA(A, d, obj_part)
            i += 1
        if stop != 0 and i == stop:
            break

    binarify(A)
    print("[TIME] : ", dt.now()-start, "for", i, "images.")
    print(A)
    return A


def display_RF(rf_center, filepath, name, iid):

    image = cv2.resize(cv2.imread(filepath), (224, 224),
                       interpolation=cv2.INTER_LINEAR)

    fx = int(rf_center[1])      # Add your Focus cordinates here
    fy = int(rf_center[0])
    sigma = RF_SIZE/2.0         # Standard Deviation of the Gaussian
    rows, cols = 224, 224

    start = (fx - int(RF_SIZE/2), fy - int(RF_SIZE/2))
    end   = (fx + int(RF_SIZE/2), fy + int(RF_SIZE/2))

    result = np.copy(image)
    result[:, :, :] = 0

    a = cv2.getGaussianKernel(2*cols, sigma)[cols-fx:2*cols-fx]
    b = cv2.getGaussianKernel(2*rows, sigma)[rows-fy:2*rows-fy]
    c = b*a.T
    d = c/c.max()
    result[:, :, 0] = image[:, :, 0]*d
    result[:, :, 1] = image[:, :, 1]*d
    result[:, :, 2] = image[:, :, 2]*d
    result = cv2.resize(cv2.rectangle(result, start, end, (0, 0, 255)), (300, 300), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(name, result)
    cv2.imwrite(name+iid, result)


def visualize_objpart_RF(m_trained, test_image, A, filepath, iid):
    
    max_pool_model = Model(inputs=m_trained.input,outputs=m_trained.get_layer("final_max_pool").output)
    pool_output = max_pool_model.predict(test_image)
    x = tf.reshape(pool_output, shape=(7,7,512))
    heados = []
    torsos = []
    legos  = []
    tailos = []
    for i in range(512):
        heados.append(A[i][HEAD_PARTS])
        torsos.append(A[i][TORSO_PARTS])
        legos.append( A[i][LEG_PARTS])
        tailos.append(A[i][TAIL_PARTS])
    
    x_heados = tf.reduce_mean(tf.multiply(heados, x), axis=2)
    x_torsos = tf.reduce_mean(tf.multiply(torsos, x), axis=2)
    x_legos  = tf.reduce_mean(tf.multiply(legos,  x), axis=2)
    x_tailos = tf.reduce_mean(tf.multiply(tailos, x), axis=2)

    heados_i = tf.math.argmax(tf.reduce_max(x_heados, axis=1),output_type=tf.int32).numpy()
    heados_j = tf.math.argmax(tf.reduce_max(x_heados, axis=0),output_type=tf.int32).numpy()
    heados_center, size = receptive_field((heados_j, heados_i))

    torsos_i = tf.math.argmax(tf.reduce_max(x_torsos, axis=1),output_type=tf.int32).numpy()
    torsos_j = tf.math.argmax(tf.reduce_max(x_torsos, axis=0),output_type=tf.int32).numpy()
    torsos_center, size = receptive_field((torsos_j, torsos_i))

    legos_i  = tf.math.argmax(tf.reduce_max(x_legos,  axis=1),output_type=tf.int32).numpy()
    legos_j  = tf.math.argmax(tf.reduce_max(x_legos,  axis=0),output_type=tf.int32).numpy()
    legos_center, size = receptive_field((legos_j, legos_i))

    tailos_i = tf.math.argmax(tf.reduce_max(x_tailos, axis=1),output_type=tf.int32).numpy()
    tailos_j = tf.math.argmax(tf.reduce_max(x_tailos, axis=0),output_type=tf.int32).numpy()
    tailos_center, size = receptive_field((tailos_j, tailos_i))

    original = cv2.resize(cv2.imread(filepath), (300,300), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('original', original)
    cv2.imwrite('original'+iid, original)
    display_RF(heados_center, filepath, 'head' , iid)
    display_RF(torsos_center, filepath, 'torso', iid)
    display_RF(legos_center,  filepath, 'legs' , iid)
    display_RF(tailos_center, filepath, 'tail' , iid)

    i = tf.math.argmax(tf.reduce_max(tf.reduce_mean(x, axis=2), axis=1), output_type=tf.int32).numpy()
    j = tf.math.argmax(tf.reduce_max(tf.reduce_mean(x, axis=2), axis=0), output_type=tf.int32).numpy()

    center, size = receptive_field((j,i))
    display_RF(center, filepath, 'all', iid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
