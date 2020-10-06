import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import img_to_array
from classes.maskLayer import MaskLayer
from utils.dataset_utils import load_test_image
from utils.receptvie_field import receptive_field

RF_SIZE = 54
MODELS  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MASKED1 = os.path.join(MODELS, "masked1_no_dropout_binary_50_epochs_24_9_2020_14_7.h5")
POS_IMAGE_SET_TEST = "./dataset/train_val/test/bird/"

HEAD_PART   = 0
TORSO_PART  = 1
LEG_PART    = 2
TAIL_PART   = 3

parts = {'HEAD' :['head', 'beak', 'leye', 'reye'],
         'TORSO':['torso', 'neck', 'lwing', 'rwing'],
         'LEG'  :['lleg', 'rleg', 'lfoot', 'rfoot'],
         'TAIL' :['tail']}

def display_RF(rf_center):
    boh = np.zeros(shape=(224, 224, 512), dtype=np.uint8)

    for i in range(224):
        for j in range(224):
            if (i >= rf_center[0]-(RF_SIZE/2) and i <= rf_center[0]+(RF_SIZE/2)) and (j >= rf_center[1]-(RF_SIZE/2) and j <= rf_center[1]+(RF_SIZE/2)):
                boh[i, j, d] = 1

    tens_boh = boh[:,:,d]
    image = cv2.resize(cv2.imread(POS_IMAGE_SET_TEST+img), (224,224))
    masked_image = cv2.bitwise_and(image,image,mask=tens_boh)

    name, boxes = read_content(img[:-4])
    print(name)
    print(boxes)

    cv2.imshow("Falcone (non Giovanni)", masked_image)
    #cv2.imshow("Falcone (not Giovanni)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#with tf.device("/CPU:0"):
m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})

max_pool_model = Model(inputs=m_trained.input, outputs=m_trained.get_layer("final_max_pool").output)

A = []
for img in os.listdir(POS_IMAGE_SET_TEST):
    if img.endswith('.jpg'):
        test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)
        
        pool_output = max_pool_model.predict(test_image)

        rows_idx = tf.math.argmax(tf.reduce_max(pool_output[0], axis=1), output_type=tf.int32)
        cols_idx = tf.math.argmax(tf.reduce_max(pool_output[0], axis=0), output_type=tf.int32)

        for d in range(len(rows_idx)):
            max_i = rows_idx[d].numpy()
            max_j = cols_idx[d].numpy()

            rf_center, rf_size = receptive_field((max_i,max_j))
            
            for f in range(512):
                mindist = None
                annotations = load_annotation(img) 
                for a in annotations:
                    aux = dist(rf_center, a.center)
                    if (mindist is None and aux < threshold) or (mindist is not None and aux < mindist):
                        mindist = aux
                        obj_part = a.tag
                
                if obj_part in parts['HEAD']: 
                    A[f, HEAD_PART] += 1
                elif obj_part in parts['TORSO']:
                    A[f, TORSO_PART] += 1
                elif obj_part in parts['LEG']:
                    A[f, LEG_PART] += 1
                elif obj_part in parts['TAIL']:
                    A[f, TAIL_PART] += 1
                else:
                    pass

            break
    break

for f in range(512):
    l = argmax(A[f])
    for i in range(4):
        if i != l:
            A[f, i] = 0 

print(A)