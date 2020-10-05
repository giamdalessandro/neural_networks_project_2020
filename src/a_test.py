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



with tf.device("/CPU:0"):
    m_trained = tf.keras.models.load_model(MASKED1, custom_objects={"MaskLayer":MaskLayer()})

max_pool_model = Model(inputs=m_trained.input, outputs=m_trained.get_layer("final_max_pool").output)

x_list = []
for img in os.listdir(POS_IMAGE_SET_TEST):
    if img.endswith('.jpg'):
        test_image = load_test_image(folder=POS_IMAGE_SET_TEST, fileid=img)

        #plt.imshow(image)
        #plt.show()

        pool_output = max_pool_model.predict(test_image)
        x_list.append(pool_output[0])

        rows_idx = tf.math.argmax(tf.reduce_max(pool_output[0], axis=1), output_type=tf.int32)
        cols_idx = tf.math.argmax(tf.reduce_max(pool_output[0], axis=0), output_type=tf.int32)

        boh = np.zeros(shape=(224,224,512), dtype=np.uint8)
        for d in range(len(rows_idx)):
            max_i = rows_idx[d].numpy()
            max_j = cols_idx[d].numpy()

            rf_center, rf_size = receptive_field((max_i,max_j))
            #print(rf_center)
            for i in range(224):
                for j in range(224):
                    if (i >= rf_center[0]-(RF_SIZE/2) and i <= rf_center[0]+(RF_SIZE/2)) and (j >= rf_center[1]-(RF_SIZE/2) and j <= rf_center[1]+(RF_SIZE/2)):
                        boh[i,j,d] = 1

            
            tens_boh = boh[:,:,d]
            image = cv2.resize(cv2.imread(POS_IMAGE_SET_TEST+img), (224,224))
            masked_image = cv2.bitwise_and(image,image,mask=tens_boh)

            cv2.imshow("Falcone (non Giovanni)", masked_image)
            #cv2.imshow("Falcone (not Giovanni)", image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

            #break
        break

'''
import xml.etree.ElementTree as ET   # phone home.

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


name, boxes = read_content(
    "/media/luca/DATA2/uni/neural_networks_project_2020/dataset/pascalvocpart/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/Annotations/2007_000027.xml")
print(name)
print(boxes)'''